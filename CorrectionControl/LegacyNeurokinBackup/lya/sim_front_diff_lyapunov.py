from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Geometry:
    front_track_a: float = 0.32
    front_to_rear_b: float = 0.42
    caster_limit_deg: float = 100.0


@dataclass
class Gains:
    kx: float = 2.0
    ky: float = 4.0
    kth: float = 2.0


@dataclass
class State:
    x: float
    y: float
    theta: float


@dataclass
class Reference:
    x: float
    y: float
    theta: float
    v: float
    omega: float


def circle_reference(t: float, radius: float = 1.5, speed: float = 0.35) -> Reference:
    omega = speed / radius
    phi = omega * t
    x = radius * math.cos(phi)
    y = radius * math.sin(phi)
    theta = wrap_pi(phi + math.pi / 2.0)
    return Reference(x=x, y=y, theta=theta, v=speed, omega=omega)


def tracking_error(state: State, ref: Reference) -> tuple[float, float, float]:
    dx = ref.x - state.x
    dy = ref.y - state.y
    c = math.cos(state.theta)
    s = math.sin(state.theta)
    ex = c * dx + s * dy
    ey = -s * dx + c * dy
    eth = wrap_pi(ref.theta - state.theta)
    return ex, ey, eth


def lyapunov_control(state: State, ref: Reference, gains: Gains) -> tuple[float, float, tuple[float, float, float]]:
    ex, ey, eth = tracking_error(state, ref)
    v = ref.v * math.cos(eth) + gains.kx * ex
    omega = ref.omega + gains.ky * ref.v * ey + gains.kth * math.sin(eth)
    return v, omega, (ex, ey, eth)


def front_wheel_speeds(v: float, omega: float, geom: Geometry) -> tuple[float, float]:
    vr = v + 0.5 * geom.front_track_a * omega
    vl = v - 0.5 * geom.front_track_a * omega
    return vl, vr


def rear_caster_angle(v: float, omega: float, geom: Geometry) -> float:
    return math.atan2(-geom.front_to_rear_b * omega, v)


def step_unicycle(state: State, v: float, omega: float, dt: float) -> State:
    return State(
        x=state.x + v * math.cos(state.theta) * dt,
        y=state.y + v * math.sin(state.theta) * dt,
        theta=wrap_pi(state.theta + omega * dt),
    )


def simulate() -> list[dict[str, float]]:
    geom = Geometry()
    gains = Gains()
    dt = 0.02
    total_time = 30.0
    state = State(x=0.4, y=-0.6, theta=math.radians(40.0))
    rows: list[dict[str, float]] = []

    steps = int(total_time / dt)
    for i in range(steps + 1):
        t = i * dt
        ref = circle_reference(t)
        v, omega, (ex, ey, eth) = lyapunov_control(state, ref, gains)

        gamma = rear_caster_angle(v, omega, geom)
        gamma_limit = math.radians(geom.caster_limit_deg)
        caster_ok = abs(gamma) <= gamma_limit

        vl, vr = front_wheel_speeds(v, omega, geom)
        rows.append(
            {
                "t": t,
                "x": state.x,
                "y": state.y,
                "theta": state.theta,
                "x_ref": ref.x,
                "y_ref": ref.y,
                "theta_ref": ref.theta,
                "ex": ex,
                "ey": ey,
                "eth": eth,
                "v_cmd": v,
                "omega_cmd": omega,
                "front_left_speed": vl,
                "front_right_speed": vr,
                "rear_caster_angle_deg": math.degrees(gamma),
                "caster_ok": float(caster_ok),
                "V": 0.5 * (ex * ex + ey * ey) + (1.0 / gains.ky) * (1.0 - math.cos(eth)),
                "Vdot_theory": -gains.kx * ex * ex - (gains.kth / gains.ky) * (math.sin(eth) ** 2),
            }
        )
        state = step_unicycle(state, v, omega, dt)
    return rows


def plot_results(rows: list[dict[str, float]], out_dir: Path) -> None:
    t = [row["t"] for row in rows]

    plt.figure(figsize=(7.2, 6.0), dpi=150)
    plt.plot([row["x_ref"] for row in rows], [row["y_ref"] for row in rows], "--", label="reference")
    plt.plot([row["x"] for row in rows], [row["y"] for row in rows], label="actual")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Front-Axle Tracking Trajectory")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "front_diff_trajectory.png")
    plt.close()

    plt.figure(figsize=(8.0, 5.0), dpi=150)
    plt.plot(t, [row["ex"] for row in rows], label="$e_x$")
    plt.plot(t, [row["ey"] for row in rows], label="$e_y$")
    plt.plot(t, [math.degrees(row["eth"]) for row in rows], label="$e_\\theta$ [deg]")
    plt.grid(True, alpha=0.3)
    plt.xlabel("time [s]")
    plt.title("Tracking Errors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "front_diff_tracking_errors.png")
    plt.close()

    plt.figure(figsize=(8.0, 5.0), dpi=150)
    plt.plot(t, [row["V"] for row in rows], label="$V$")
    plt.plot(t, [row["Vdot_theory"] for row in rows], label="$\\dot{V}$ theory")
    plt.grid(True, alpha=0.3)
    plt.xlabel("time [s]")
    plt.title("Lyapunov Function and Theoretical Derivative")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "front_diff_lyapunov.png")
    plt.close()

    plt.figure(figsize=(8.0, 5.0), dpi=150)
    plt.plot(t, [row["front_left_speed"] for row in rows], label="front left speed")
    plt.plot(t, [row["front_right_speed"] for row in rows], label="front right speed")
    plt.grid(True, alpha=0.3)
    plt.xlabel("time [s]")
    plt.ylabel("speed [m/s]")
    plt.title("Front Wheel Speeds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "front_diff_wheel_speeds.png")
    plt.close()

    plt.figure(figsize=(8.0, 5.0), dpi=150)
    gamma = [row["rear_caster_angle_deg"] for row in rows]
    plt.plot(t, gamma, label="rear caster angle")
    plt.axhline(100.0, color="red", linestyle="--", linewidth=1.0, label="+100 deg limit")
    plt.axhline(-100.0, color="red", linestyle="--", linewidth=1.0, label="-100 deg limit")
    plt.grid(True, alpha=0.3)
    plt.xlabel("time [s]")
    plt.ylabel("angle [deg]")
    plt.title("Rear Passive Caster Angle")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "front_diff_caster_angle.png")
    plt.close()


def _box_arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="->",
            mutation_scale=14,
            linewidth=1.6,
            color="#303030",
        )
    )


def _flow_box(ax, xy: tuple[float, float], width: float, height: float, text: str, face: str) -> None:
    x, y = xy
    ax.add_patch(
        Rectangle(
            (x, y),
            width,
            height,
            linewidth=1.2,
            edgecolor="#303030",
            facecolor=face,
        )
    )
    ax.text(
        x + width / 2.0,
        y + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.25,
    )


def plot_geometry_and_flows(out_dir: Path) -> None:
    geom = Geometry()
    a = geom.front_track_a
    b = geom.front_to_rear_b
    v_example = 0.45
    omega_example = 0.75
    gamma = rear_caster_angle(v_example, omega_example, geom)

    plt.figure(figsize=(8.0, 4.8), dpi=150)
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.axis("off")
    front_center = (0.0, 0.0)
    rear = (-b, 0.0)
    cog = (-b / 2.0, 0.0)
    left_front = (0.0, a / 2.0)
    right_front = (0.0, -a / 2.0)

    ax.plot([-b, 0.08], [0.0, 0.0], color="#202020", linewidth=2.0)
    ax.plot([0.0, 0.0], [-a / 2.0 - 0.07, a / 2.0 + 0.07], color="#202020", linewidth=2.0)
    ax.add_patch(Rectangle((-0.03, a / 2.0 - 0.055), 0.06, 0.11, angle=0, color="#1f77b4"))
    ax.add_patch(Rectangle((-0.03, -a / 2.0 - 0.055), 0.06, 0.11, angle=0, color="#ff7f0e"))
    ax.plot(rear[0], rear[1], "ko", markersize=6)
    caster_len = 0.16
    ax.plot(
        [rear[0] - 0.5 * caster_len * math.cos(gamma), rear[0] + 0.5 * caster_len * math.cos(gamma)],
        [rear[1] - 0.5 * caster_len * math.sin(gamma), rear[1] + 0.5 * caster_len * math.sin(gamma)],
        color="#2ca02c",
        linewidth=4.0,
    )
    ax.plot(cog[0], cog[1], "o", color="#d62728", markersize=7)

    _box_arrow(ax, front_center, (0.28, 0.0))
    _box_arrow(ax, rear, (rear[0] + 0.18, rear[1] - b * omega_example / v_example * 0.12))
    ax.text(0.30, 0.015, "$x_b$, heading $\\theta$", fontsize=10)
    ax.text(-0.25, 0.12, "$a$", fontsize=11, bbox={"facecolor": "white", "edgecolor": "#cccccc"})
    ax.text(-b / 2.0, -0.075, "$b$", fontsize=11, bbox={"facecolor": "white", "edgecolor": "#cccccc"})
    ax.text(left_front[0] + 0.035, left_front[1], "$v_L$", fontsize=11, va="center")
    ax.text(right_front[0] + 0.035, right_front[1], "$v_R$", fontsize=11, va="center")
    ax.text(front_center[0] + 0.02, front_center[1] + 0.02, "$P$", fontsize=12)
    ax.text(cog[0] - 0.03, cog[1] + 0.035, "$C$", fontsize=12, color="#d62728")
    ax.text(rear[0] - 0.15, rear[1] - 0.04, "passive caster\n$\\gamma=\\mathrm{atan2}(-b\\omega,v)$", fontsize=9)
    ax.text(
        -0.52,
        0.29,
        "Front differential drive:\n$v=(v_R+v_L)/2$\n$\\omega=(v_R-v_L)/a$",
        fontsize=10,
        bbox={"facecolor": "#f7f7f7", "edgecolor": "#303030", "boxstyle": "round,pad=0.35"},
    )
    ax.set_xlim(-0.72, 0.42)
    ax.set_ylim(-0.34, 0.45)
    ax.set_title("Vehicle Geometry Used by the Lyapunov Controller")
    plt.tight_layout()
    plt.savefig(out_dir / "front_diff_geometry.png")
    plt.close()

    plt.figure(figsize=(10.2, 4.0), dpi=150)
    ax = plt.gca()
    ax.axis("off")
    boxes = [
        ((0.2, 1.9), 1.55, 0.85, "Reference\n$(x_d,y_d,\\theta_d,v_d,\\omega_d)$", "#e8f1fb"),
        ((0.2, 0.65), 1.55, 0.85, "Measured state\n$(x,y,\\theta)$", "#f5f5f5"),
        ((2.25, 1.25), 1.7, 0.9, "Body-frame\ntracking error\n$(e_x,e_y,e_\\theta)$", "#fff4df"),
        ((4.55, 1.25), 1.8, 0.9, "Lyapunov law\n$v,\\omega$", "#e6f4ea"),
        ((6.95, 1.25), 1.75, 0.9, "Wheel map\n$v_L,v_R$", "#fce8e6"),
        ((9.2, 1.25), 1.55, 0.9, "Vehicle\nkinematics", "#eeeeff"),
    ]
    for box in boxes:
        _flow_box(ax, *box)
    _box_arrow(ax, (1.75, 2.32), (2.25, 1.88))
    _box_arrow(ax, (1.75, 1.08), (2.25, 1.52))
    _box_arrow(ax, (3.95, 1.70), (4.55, 1.70))
    _box_arrow(ax, (6.35, 1.70), (6.95, 1.70))
    _box_arrow(ax, (8.70, 1.70), (9.20, 1.70))
    _box_arrow(ax, (9.95, 1.25), (1.05, 0.65))
    ax.text(
        6.7,
        0.35,
        "Feasibility monitor: wheel speed limits and $|\\gamma|\\leq 100^\\circ$",
        fontsize=10,
        bbox={"facecolor": "#f7f7f7", "edgecolor": "#303030", "boxstyle": "round,pad=0.35"},
    )
    ax.set_xlim(0.0, 11.0)
    ax.set_ylim(0.0, 3.2)
    ax.set_title("Closed-Loop Control Flow")
    plt.tight_layout()
    plt.savefig(out_dir / "front_diff_control_flow.png")
    plt.close()

    plt.figure(figsize=(9.4, 4.8), dpi=150)
    ax = plt.gca()
    ax.axis("off")
    proof_boxes = [
        ((0.25, 2.95), 2.0, 0.8, "Assume kinematic model\n$\\dot{x}=v\\cos\\theta$\n$\\dot{y}=v\\sin\\theta$"),
        ((3.0, 2.95), 2.0, 0.8, "Derive error dynamics\n$\\dot e_x,\\dot e_y,\\dot e_\\theta$"),
        ((5.75, 2.95), 2.0, 0.8, "Choose candidate\n$V=\\frac{1}{2}(e_x^2+e_y^2)$\n$+\\frac{1}{k_y}(1-\\cos e_\\theta)$"),
        ((5.75, 1.25), 2.0, 0.8, "Substitute control\n$v=v_d\\cos e_\\theta+k_xe_x$\n$\\omega=\\omega_d+k_yv_de_y+k_\\theta\\sin e_\\theta$"),
        ((3.0, 1.25), 2.0, 0.8, "Cancel cross terms\n$v_de_y\\sin e_\\theta$"),
        ((0.25, 1.25), 2.0, 0.8, "Obtain\n$\\dot V=-k_xe_x^2$\n$-\\frac{k_\\theta}{k_y}\\sin^2 e_\\theta\\leq 0$"),
        ((3.0, 0.0), 2.0, 0.8, "With $v_d>0$ and smoothness:\n$e_x,e_y,e_\\theta\\to0$"),
    ]
    for xy, width, height, text in proof_boxes:
        _flow_box(ax, xy, width, height, text, "#f7f7f7")
    _box_arrow(ax, (2.25, 3.35), (3.0, 3.35))
    _box_arrow(ax, (5.0, 3.35), (5.75, 3.35))
    _box_arrow(ax, (6.75, 2.95), (6.75, 2.05))
    _box_arrow(ax, (5.75, 1.65), (5.0, 1.65))
    _box_arrow(ax, (3.0, 1.65), (2.25, 1.65))
    _box_arrow(ax, (1.25, 1.25), (3.0, 0.40))
    ax.set_xlim(0.0, 8.1)
    ax.set_ylim(-0.15, 3.95)
    ax.set_title("Lyapunov Proof Flow")
    plt.tight_layout()
    plt.savefig(out_dir / "front_diff_proof_flow.png")
    plt.close()


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    rows = simulate()
    out_csv = out_dir / "front_diff_lyapunov_sim.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_results(rows, out_dir)
    plot_geometry_and_flows(out_dir)

    final = rows[-1]
    print(f"wrote {out_csv}")
    print(
        "final error: "
        f"ex={final['ex']:.4f}, ey={final['ey']:.4f}, eth={math.degrees(final['eth']):.2f} deg"
    )
    print(
        "max abs caster angle: "
        f"{max(abs(row['rear_caster_angle_deg']) for row in rows):.2f} deg"
    )
    print("wrote visualization PNGs in", out_dir)


if __name__ == "__main__":
    main()
