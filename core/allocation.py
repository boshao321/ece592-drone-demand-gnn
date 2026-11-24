"""Stage 2 of the predict-then-optimize pipeline: fleet allocation over the
sites selected by the order-flow network stage.

Design notes, and why each choice is made this way.

The siting stage fixes WHERE aircraft can launch; it is an infrastructure
decision and is made from historical order-flow structure, not from a one-hour
forecast. This stage fixes HOW MANY aircraft stand at each site in each 15
minute slot, which is the decision a forecast can actually inform, because
repositioning an aircraft takes lead time. That is why the allocation consumes
the t+60 forecast rather than t+15: at the moment the decision must be taken,
t+60 is the horizon whose demand is still unobserved.

Capacity is counted in aircraft-minutes rather than in sorties, because sortie
duration varies with the origin-destination distance actually observed in the
data. An order o launched from site s occupies

    tau_o = 2 * d_o / v_cruise + t_service

aircraft-minutes, the factor two covering the return leg to the launch site.

Given an estimate of required aircraft-minutes at each site, the allocation that
maximizes served minutes is a separable concave problem, so the exact optimum is
obtained by assigning aircraft one at a time to whichever site still has unmet
estimated demand. No solver is required, and the policy comparison is therefore
about the quality of the estimate rather than about solver differences.

Four estimates are compared, which is the standard benchmark set for a
predict-then-optimize pipeline: a static policy using the historical mean for
that time of day, a lagged policy using the most recent observation available
given the lead time, the model forecast, and perfect foresight as an upper
bound. Reporting the oracle is what turns a raw served-demand number into a
statement about how much of the attainable gap the forecast actually closes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ------------------------------------------------------ UAV operating envelope
# A light multirotor logistics platform. The cruise speed and service overhead
# set sortie duration; the service radius sets which demand grids a site can
# reach. Values are swept in the sensitivity analysis rather than assumed exact.
V_CRUISE_KMH = 60.0
T_SERVICE_MIN = 4.0          # loading, handover, battery swap per sortie
SERVICE_RADIUS_KM = 3.0      # grid to launch site
MAX_SORTIE_RANGE_KM = 8.0    # one-way distance an order may require
SLOT_MINUTES = 15.0


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def sortie_minutes(dist_km, v_kmh=V_CRUISE_KMH, t_service=T_SERVICE_MIN):
    return 2.0 * dist_km / v_kmh * 60.0 + t_service


def tour_minutes(d_site_pickup, d_od, d_dropoff_site,
                 v_kmh=V_CRUISE_KMH, t_service=T_SERVICE_MIN):
    """Closed tour, site to pickup to dropoff and back to the site.

    The default cost model places the pad at the pickup and therefore prices the
    pad-to-demand leg at zero. That is adequate while the question is how many
    aircraft to send to a fixed set of pads, but it leaves capacity insensitive
    to where those pads are, so a comparison between site sets needs the full
    tour."""
    return (d_site_pickup + d_od + d_dropoff_site) / v_kmh * 60.0 + t_service


HIGH_VALUE_THRESHOLD = 53.90     # top quartile of order value on days 1-6; the scripts
                                 # overwrite it with data.high_value_threshold_of(df)


@dataclass
class Scenario:
    """Everything the allocation needs, all defined on one node set."""
    site_idx: np.ndarray          # indices (into the node set) of selected sites
    grid_to_site: np.ndarray      # for each grid, the site index it is assigned to, or -1
    req_minutes: np.ndarray       # [T_slots, n_sites] aircraft-minutes for ALL servable orders
    req_minutes_hv: np.ndarray    # [T_slots, n_sites] aircraft-minutes for high-value orders
    req_orders: np.ndarray        # [T_slots, n_sites] realized servable order count
    req_orders_hv: np.ndarray     # [T_slots, n_sites] realized high-value order count
    req_value: np.ndarray         # [T_slots, n_sites] realized servable order value
    order_book: dict              # (slot, site) -> array of (tau, price), value-sorted
    slots: np.ndarray             # slot index of each row
    n_sites: int


def assign_grids_to_sites(coords, site_idx, radius_km=SERVICE_RADIUS_KM):
    """Each demand grid is served by its nearest selected site inside the
    service radius. Grids with no site in range are outside the aerial network
    and are reported separately rather than silently dropped."""
    site_coords = coords[site_idx]
    d = haversine_km(coords[:, 0][:, None], coords[:, 1][:, None],
                     site_coords[:, 0][None, :], site_coords[:, 1][None, :])
    nearest = d.argmin(axis=1)
    within = d.min(axis=1) <= radius_km
    out = np.where(within, nearest, -1)
    return out, d


def build_scenario(orders, coords, site_idx, slots, radius_km=SERVICE_RADIUS_KM,
                   v_kmh=V_CRUISE_KMH, t_service=T_SERVICE_MIN,
                   max_range_km=MAX_SORTIE_RANGE_KM, grid_index=None,
                   positioning=False):
    """Turn raw orders into per-slot, per-site aircraft-minute requirements.

    With positioning=False the sortie cost is 2*d_od/v + t_service, the model
    used throughout the allocation stage. With positioning=True the cost is the
    full closed tour from the launch site, which is what makes the cost of a
    sortie depend on where its site was placed."""
    grid_to_site, _ = assign_grids_to_sites(coords, site_idx, radius_km)
    n_sites = len(site_idx)
    slot_pos = {s: i for i, s in enumerate(slots)}

    o = orders.copy()
    o["pickup_node"] = [grid_index.get(k, -1) for k in
                        zip(o["p_lat_snap"], o["p_lng_snap"])]
    o = o[o["pickup_node"] >= 0]
    o["site"] = grid_to_site[o["pickup_node"].to_numpy()]
    o = o[o["site"] >= 0]
    o = o[o["od_km"] <= max_range_km]
    o = o[o["slot"].isin(slot_pos)]
    if positioning:
        st = coords[np.asarray(site_idx)][o["site"].to_numpy()]
        d_sp = haversine_km(st[:, 0], st[:, 1],
                            o["pickup_latitude"].to_numpy(), o["pickup_longitude"].to_numpy())
        d_ds = haversine_km(o["dropoff_latitude"].to_numpy(), o["dropoff_longitude"].to_numpy(),
                            st[:, 0], st[:, 1])
        o["tau"] = tour_minutes(d_sp, o["od_km"].to_numpy(), d_ds, v_kmh, t_service)
    else:
        o["tau"] = sortie_minutes(o["od_km"].to_numpy(), v_kmh, t_service)

    T = len(slots)
    req_minutes = np.zeros((T, n_sites))
    req_minutes_hv = np.zeros((T, n_sites))
    req_orders = np.zeros((T, n_sites))
    req_orders_hv = np.zeros((T, n_sites))
    req_value = np.zeros((T, n_sites))
    order_book = {}
    for (sl, si), g in o.groupby(["slot", "site"]):
        i = slot_pos[int(sl)]
        j = int(si)
        arr = np.stack([g["tau"].to_numpy(), g["package_price"].to_numpy()], axis=1)
        arr = arr[np.argsort(-arr[:, 1])]          # highest value first
        order_book[(i, j)] = arr
        hv = arr[:, 1] >= HIGH_VALUE_THRESHOLD
        req_minutes[i, j] = arr[:, 0].sum()
        req_minutes_hv[i, j] = arr[hv, 0].sum()
        req_orders[i, j] = len(arr)
        req_orders_hv[i, j] = int(hv.sum())
        req_value[i, j] = arr[:, 1].sum()

    return Scenario(np.asarray(site_idx), grid_to_site, req_minutes, req_minutes_hv,
                    req_orders, req_orders_hv, req_value, order_book,
                    np.asarray(slots), n_sites)


def allocate(estimate_minutes, fleet_size):
    """Exact optimum of  max sum_s min(15 * x_s, est_s)  s.t. sum_s x_s <= M.

    The objective is separable and concave in each x_s, so assigning aircraft
    one at a time to the site with the largest remaining unmet estimate is
    optimal. In closed form, every site first receives its full integer requirement
    when the fleet suffices; otherwise sites are filled whole in descending order
    of estimated demand until the fleet runs out.
    """
    T, S = estimate_minutes.shape
    x = np.zeros((T, S), dtype=int)
    for t in range(T):
        est = estimate_minutes[t]
        need = np.ceil(np.maximum(est, 0) / SLOT_MINUTES).astype(int)
        if need.sum() <= fleet_size:
            x[t] = need
            continue
        # rank sites by estimated demand and fill greedily; within a site the
        # marginal return of an extra aircraft is constant until its estimate is
        # met, so filling whole sites in descending order is optimal
        order = np.argsort(-est)
        remaining = fleet_size
        for j in order:
            take = min(need[j], remaining)
            x[t, j] = take
            remaining -= take
            if remaining <= 0:
                break
    return x


def evaluate(x, sc: Scenario):
    """Serve the realized orders with the allocated capacity, highest value
    first, and report what the fleet actually achieved."""
    T, S = x.shape
    served_orders = 0.0
    served_value = 0.0
    served_minutes = 0.0
    total_orders = sc.req_orders.sum()
    total_value = sc.req_value.sum()
    hv_threshold = HIGH_VALUE_THRESHOLD
    served_hv = 0.0
    total_hv = sc.req_orders_hv.sum()
    for (t, j), arr in sc.order_book.items():
        cap = x[t, j] * SLOT_MINUTES
        if cap <= 0:
            continue
        cum = np.cumsum(arr[:, 0])
        k = int(np.searchsorted(cum, cap, side="right"))
        served_orders += k
        served_value += arr[:k, 1].sum()
        served_minutes += cum[k - 1] if k > 0 else 0.0
        served_hv += (arr[:k, 1] >= hv_threshold).sum()
    capacity_minutes = x.sum() * SLOT_MINUTES
    return {
        "served_orders": served_orders,
        "served_order_share_pct": 100.0 * served_orders / total_orders if total_orders else np.nan,
        "served_high_value_orders": served_hv,
        "served_high_value_share_pct": 100.0 * served_hv / total_hv if total_hv else np.nan,
        "served_value": served_value,
        "served_value_share_pct": 100.0 * served_value / total_value if total_value else np.nan,
        "unmet_orders": total_orders - served_orders,
        "fleet_utilization_pct": 100.0 * served_minutes / capacity_minutes if capacity_minutes else np.nan,
        "aircraft_slots_used": int(x.sum()),
    }


def policy_estimates(sc: Scenario, demand_full, node_to_site, pred_t60,
                     train_slots, minutes_per_order, lead_slots: int = 4):
    """Aircraft-minute estimates under each policy.

    Every policy estimates the same quantity, the aircraft-minutes needed to
    serve the high-value orders at each site, because that is the quantity the
    objective rewards. Estimating total demand instead and then being scored on
    high-value service is a target mismatch, and the experiment reports what
    that mismatch costs.

    static    historical mean for that time of day, from the training days only
    lagged    the most recent observation available given a 60 minute lead time
    forecast  model prediction for t+60, converted to minutes
    oracle    the realized requirement, i.e. perfect foresight
    """
    T, S = sc.req_minutes.shape
    slots = sc.slots
    est = {}

    # static: per site, per time-of-day, mean required minutes over training days
    site_demand = np.zeros((demand_full.shape[0], S))
    for node in range(demand_full.shape[1]):
        j = node_to_site[node]
        if j >= 0:
            site_demand[:, j] += demand_full[:, node]
    tod_all = np.arange(demand_full.shape[0]) % 96
    static = np.zeros((T, S))
    for i, sl in enumerate(slots):
        m = (tod_all == sl % 96) & np.isin(np.arange(demand_full.shape[0]), train_slots)
        static[i] = site_demand[m].mean(axis=0) if m.any() else 0.0
    est["static"] = static * minutes_per_order

    # lagged: the most recent observation available at decision time, which is
    # `lead_slots` slots earlier. The lead time is a property of the operation,
    # not of the model, so it moves together with the forecast horizon used.
    lagged = np.zeros((T, S))
    for i, sl in enumerate(slots):
        src = int(sl) - lead_slots
        lagged[i] = site_demand[src] if src >= 0 else 0.0
    est["lagged"] = lagged * minutes_per_order

    # forecast: predicted order counts at t+60, aggregated to sites
    est["forecast"] = pred_t60 * minutes_per_order

    # oracle: exactly the minutes needed to clear the high-value orders
    est["oracle"] = sc.req_minutes_hv.copy()
    return est


def nodes_to_sites(pred_nodes, node_to_site, n_sites):
    """Aggregate a per-grid forecast to per-site, using the same assignment the
    optimizer uses. This is the only place the two stages exchange data, and it
    is why both must be defined on the same node set."""
    T = pred_nodes.shape[0]
    out = np.zeros((T, n_sites))
    for node in range(pred_nodes.shape[1]):
        j = node_to_site[node]
        if j >= 0:
            out[:, j] += np.maximum(pred_nodes[:, node], 0.0)
    return out
