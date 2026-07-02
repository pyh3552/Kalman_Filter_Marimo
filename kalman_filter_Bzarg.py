import marimo

__generated_with = "0.23.10"
app = marimo.App(width="medium", app_title="卡尔曼滤波：从不确定性到信息融合")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import re
    from io import StringIO
    from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, ConnectionPatch
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Rectangle, Circle, Polygon
    from matplotlib.gridspec import GridSpec
    import matplotlib.patheffects as pe


    # SVG 保留文本节点，让浏览器使用系统中文字体渲染，避免 WASM 中读取本地字体文件。
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Microsoft YaHei",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    SVG_FONT_FAMILY = (
        "'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', "
        "'Noto Sans CJK SC', 'Arial Unicode MS', 'DejaVu Sans', sans-serif"
    )

    def prefer_browser_cjk_fonts(svg):
        svg = re.sub(
            r"font-family:\s*[^;\"]+",
            f"font-family: {SVG_FONT_FAMILY}",
            svg,
        )
        svg_font_css = (
            "<style type=\"text/css\">"
            f"text,tspan{{font-family:{SVG_FONT_FAMILY} !important;}}"
            "</style>"
        )
        return svg.replace("<defs>", f"<defs>{svg_font_css}", 1)

    def figure_as_svg(figure):
        buffer = StringIO()
        figure.savefig(buffer, format="svg", bbox_inches="tight")
        plt.close(figure)
        return mo.Html(prefer_browser_cjk_fonts(buffer.getvalue()))

    return (
        ConnectionPatch,
        Ellipse,
        FancyArrowPatch,
        FancyBboxPatch,
        GridSpec,
        figure_as_svg,
        mo,
        np,
        pe,
        plt,
    )


@app.cell(hide_code=True)
def _(Ellipse, FancyArrowPatch, FancyBboxPatch, np, plt):
    # 全篇统一使用的颜色。它们分别代表真实状态、预测、测量、后验与噪声。
    COLORS = {
        "truth": "#172554",
        "prior": "#d946ef",
        "measurement": "#84cc16",
        "posterior": "#2563eb",
        "process": "#14b8a6",
        "control": "#f59e0b",
        "grid": "#cbd5e1",
        "muted": "#64748b",
        "surface": "#f8fafc",
    }

    def _style_axis(ax, xlabel=None, ylabel=None, title=None):
        ax.set_facecolor(COLORS["surface"])
        ax.grid(color=COLORS["grid"], alpha=0.45, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, loc="left", fontweight="bold")

    def _covariance_ellipse(
        ax,
        mean,
        covariance,
        *,
        color,
        label=None,
        n_std=2.0,
        alpha=0.18,
        linewidth=2.0,
        linestyle="-",
        zorder=3,
    ):
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
        width, height = 2 * n_std * np.sqrt(np.maximum(eigenvalues, 1e-12))
        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=angle,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
            zorder=zorder,
        )
        ax.add_patch(ellipse)
        ax.scatter(*mean, color=color, s=42, zorder=zorder + 1)
        return ellipse

    def _gaussian_pdf(x, mean, std):
        return np.exp(-0.5 * ((x - mean) / std) ** 2) / (
            std * np.sqrt(2 * np.pi)
        )

    def _gaussian_density(mean, covariance, x_grid, y_grid):
        points = np.stack([x_grid - mean[0], y_grid - mean[1]], axis=-1)
        inverse = np.linalg.inv(covariance)
        exponent = np.einsum("...i,ij,...j->...", points, inverse, points)
        normalizer = 2 * np.pi * np.sqrt(np.linalg.det(covariance))
        return np.exp(-0.5 * exponent) / normalizer

    def _constant_velocity_process_covariance(process_std, dt):
        acceleration_map = np.array([[0.5 * dt**2], [dt]])
        return process_std**2 * (acceleration_map @ acceleration_map.T)

    def simulate_robot(process_std, measurement_std, seed, steps=50, dt=1.0):
        """生成轨迹，并用标准线性卡尔曼滤波器估计位置和速度。"""
        rng = np.random.default_rng(int(seed))
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        observation = np.array([[1.0, 0.0]])
        process_covariance = _constant_velocity_process_covariance(
            process_std, dt
        )
        measurement_covariance = np.array([[measurement_std**2]])

        truth = np.zeros((steps, 2))
        measurements = np.zeros(steps)
        estimates = np.zeros((steps, 2))
        uncertainties = np.zeros((steps, 2, 2))
        gains = np.zeros(steps)

        truth[0] = np.array([0.0, 1.2])
        estimate = np.array([0.0, 0.0])
        covariance = np.diag([measurement_std**2, 4.0])

        for index in range(steps):
            if index:
                random_acceleration = rng.normal(0.0, process_std)
                control_map = np.array([0.5 * dt**2, dt])
                truth[index] = (
                    transition @ truth[index - 1]
                    + control_map * random_acceleration
                )

            measurements[index] = truth[index, 0] + rng.normal(
                0.0, measurement_std
            )

            estimate_prior = transition @ estimate
            covariance_prior = (
                transition @ covariance @ transition.T + process_covariance
            )
            innovation = measurements[index] - (observation @ estimate_prior)[0]
            innovation_covariance = (
                observation @ covariance_prior @ observation.T
                + measurement_covariance
            )
            gain = (
                covariance_prior
                @ observation.T
                @ np.linalg.inv(innovation_covariance)
            )
            estimate = estimate_prior + gain[:, 0] * innovation
            identity = np.eye(2)
            # Joseph 形式在数值上更稳定，并保持 P 为对称半正定矩阵。
            covariance = (
                (identity - gain @ observation)
                @ covariance_prior
                @ (identity - gain @ observation).T
                + gain @ measurement_covariance @ gain.T
            )

            estimates[index] = estimate
            uncertainties[index] = covariance
            gains[index] = gain[0, 0]

        time = np.arange(steps) * dt
        return time, truth, measurements, estimates, uncertainties, gains

    def plot_robot_demo(process_std, measurement_std, seed):
        time, truth, measurements, estimates, uncertainties, gains = simulate_robot(
            process_std, measurement_std, seed
        )
        position_sigma = np.sqrt(uncertainties[:, 0, 0])

        figure, axes = plt.subplots(
            2,
            1,
            figsize=(10, 7.2),
            gridspec_kw={"height_ratios": [2.2, 1.0]},
            constrained_layout=True,
        )
        position_axis, gain_axis = axes
        position_axis.plot(
            time,
            truth[:, 0],
            color=COLORS["truth"],
            linewidth=2.5,
            label="真实位置",
        )
        position_axis.scatter(
            time,
            measurements,
            color=COLORS["measurement"],
            s=18,
            alpha=0.62,
            label="GPS 测量",
            zorder=2,
        )
        position_axis.plot(
            time,
            estimates[:, 0],
            color=COLORS["posterior"],
            linewidth=2.2,
            label="滤波估计",
            zorder=3,
        )
        position_axis.fill_between(
            time,
            estimates[:, 0] - 2 * position_sigma,
            estimates[:, 0] + 2 * position_sigma,
            color=COLORS["posterior"],
            alpha=0.13,
            label="估计的 ±2σ 区间",
        )
        _style_axis(
            position_axis,
            xlabel="时间",
            ylabel="位置",
            title="机器人定位：预测与 GPS 共同约束真实轨迹",
        )
        position_axis.legend(ncol=2, frameon=False)

        gain_axis.plot(
            time,
            gains,
            color=COLORS["control"],
            linewidth=2.2,
        )
        gain_axis.fill_between(
            time, 0, gains, color=COLORS["control"], alpha=0.16
        )
        gain_axis.set_ylim(0, 1.02)
        _style_axis(
            gain_axis,
            xlabel="时间",
            ylabel="位置增益 Kp",
            title="当前测量在更新中所占的权重",
        )
        return figure

    def plot_state_distribution(mean_position, mean_velocity, std_position, std_velocity, rho):
        mean = np.array([mean_velocity, mean_position])
        covariance = np.array(
            [
                [std_velocity**2, rho * std_velocity * std_position],
                [rho * std_velocity * std_position, std_position**2],
            ]
        )
        x_values = np.linspace(mean_velocity - 4, mean_velocity + 4, 220)
        y_values = np.linspace(mean_position - 4, mean_position + 4, 220)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        density = _gaussian_density(mean, covariance, x_grid, y_grid)

        figure, axis = plt.subplots(figsize=(8.2, 6.2), constrained_layout=True)
        levels = np.linspace(density.max() * 0.04, density.max(), 12)
        axis.contourf(
            x_grid,
            y_grid,
            density,
            levels=levels,
            cmap="Blues",
            alpha=0.82,
        )
        _covariance_ellipse(
            axis,
            mean,
            covariance,
            color=COLORS["posterior"],
            label="约 95% 的状态区域",
            alpha=0.12,
        )
        axis.axvline(mean_velocity, color=COLORS["muted"], linestyle=":", alpha=0.7)
        axis.axhline(mean_position, color=COLORS["muted"], linestyle=":", alpha=0.7)
        axis.annotate(
            rf"$\mu=({mean_position:.1f},\,{mean_velocity:.1f})$",
            xy=mean,
            xytext=(12, 12),
            textcoords="offset points",
            color=COLORS["truth"],
            fontweight="bold",
        )
        axis.set_xlim(x_values.min(), x_values.max())
        axis.set_ylim(y_values.min(), y_values.max())
        _style_axis(
            axis,
            xlabel="速度 v",
            ylabel="位置 p",
            title=f"状态空间中的二维高斯分布（相关系数 ρ={rho:.2f}）",
        )
        axis.legend(frameon=False)
        return figure

    def plot_prediction(dt, acceleration, rho):
        previous_mean = np.array([0.0, 1.0])
        previous_covariance = np.array([[1.1, 0.45 * rho], [0.45 * rho, 0.45]])
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        control = np.array([0.5 * dt**2, dt]) * acceleration
        predicted_mean = transition @ previous_mean + control
        predicted_covariance = transition @ previous_covariance @ transition.T

        figure, axis = plt.subplots(figsize=(8.6, 6.1), constrained_layout=True)
        _covariance_ellipse(
            axis,
            previous_mean[::-1],
            previous_covariance[::-1, ::-1],
            color=COLORS["posterior"],
            label="上一时刻 (x_hat k-1, P k-1)",
            alpha=0.16,
        )
        _covariance_ellipse(
            axis,
            predicted_mean[::-1],
            predicted_covariance[::-1, ::-1],
            color=COLORS["prior"],
            label="预测 (x_hat k, P k)",
            alpha=0.18,
        )
        axis.annotate(
            "",
            xy=predicted_mean[::-1],
            xytext=previous_mean[::-1],
            arrowprops={
                "arrowstyle": "->",
                "color": COLORS["control"],
                "lw": 2.4,
            },
        )
        midpoint = 0.5 * (previous_mean[::-1] + predicted_mean[::-1])
        axis.text(
            midpoint[0],
            midpoint[1],
            rf"  $F\hat{{x}}+Bu$,  $a={acceleration:.1f}$",
            color=COLORS["control"],
            fontweight="bold",
        )
        all_means = np.vstack([previous_mean[::-1], predicted_mean[::-1]])
        axis.set_xlim(all_means[:, 0].min() - 3.5, all_means[:, 0].max() + 3.5)
        axis.set_ylim(all_means[:, 1].min() - 3.5, all_means[:, 1].max() + 3.5)
        _style_axis(
            axis,
            xlabel="速度 v",
            ylabel="位置 p",
            title=f"线性预测：Δt={dt:.1f}",
        )
        axis.legend(frameon=False)
        return figure

    def plot_process_noise(process_std, dt):
        prior_mean = np.array([0.0, 1.0])
        prior_covariance = np.array([[0.75, 0.20], [0.20, 0.35]])
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        predicted_mean = transition @ prior_mean
        model_covariance = transition @ prior_covariance @ transition.T
        process_covariance = _constant_velocity_process_covariance(process_std, dt)
        noisy_covariance = model_covariance + process_covariance

        figure, axis = plt.subplots(figsize=(8.6, 6.1), constrained_layout=True)
        display_mean = predicted_mean[::-1]
        _covariance_ellipse(
            axis,
            display_mean,
            model_covariance[::-1, ::-1],
            color=COLORS["prior"],
            label="仅由运动模型传播",
            alpha=0.16,
            linestyle="--",
        )
        _covariance_ellipse(
            axis,
            display_mean,
            noisy_covariance[::-1, ::-1],
            color=COLORS["process"],
            label="加入过程噪声 Q",
            alpha=0.18,
        )
        axis.set_xlim(display_mean[0] - 4.5, display_mean[0] + 4.5)
        axis.set_ylim(display_mean[1] - 4.5, display_mean[1] + 4.5)
        _style_axis(
            axis,
            xlabel="速度 v",
            ylabel="位置 p",
            title=f"未建模扰动扩大不确定性（σa={process_std:.2f}）",
        )
        axis.legend(frameon=False)
        return figure

    def plot_measurement_mapping(sensor_mode, measurement_std):
        state_mean = np.array([2.0, 0.8])
        state_covariance = np.array([[1.45, 0.72], [0.72, 0.65]])
        mode_to_matrix = {
            "只测位置": np.array([[1.0, 0.0]]),
            "只测速度": np.array([[0.0, 1.0]]),
            "同时测位置和速度": np.eye(2),
        }
        observation = mode_to_matrix[sensor_mode]
        expected = observation @ state_mean
        measurement_covariance = measurement_std**2 * np.eye(observation.shape[0])

        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), constrained_layout=True)
        state_axis, measurement_axis = axes
        _covariance_ellipse(
            state_axis,
            state_mean[::-1],
            state_covariance[::-1, ::-1],
            color=COLORS["prior"],
            label="预测状态",
            alpha=0.22,
        )
        state_axis.set_xlim(-2.5, 4.0)
        state_axis.set_ylim(-1.5, 5.5)
        _style_axis(
            state_axis,
            xlabel="速度 v",
            ylabel="位置 p",
            title="状态空间",
        )
        state_axis.legend(frameon=False)

        if observation.shape[0] == 1:
            measurement_variance = (
                observation @ state_covariance @ observation.T
                + measurement_covariance
            )[0, 0]
            standard_deviation = np.sqrt(measurement_variance)
            x_values = np.linspace(
                expected[0] - 4 * standard_deviation,
                expected[0] + 4 * standard_deviation,
                300,
            )
            density = _gaussian_pdf(x_values, expected[0], standard_deviation)
            measurement_axis.plot(
                x_values,
                density,
                color=COLORS["measurement"],
                linewidth=2.5,
            )
            measurement_axis.fill_between(
                x_values,
                density,
                color=COLORS["measurement"],
                alpha=0.18,
            )
            measurement_axis.axvline(
                expected[0],
                color=COLORS["truth"],
                linestyle="--",
                label=r"$H\hat{x}^{-}$",
            )
            measurement_axis.set_ylim(bottom=0)
            _style_axis(
                measurement_axis,
                xlabel="传感器读数 z",
                ylabel="概率密度",
                title=f"测量空间：{sensor_mode}",
            )
            measurement_axis.legend(frameon=False)
        else:
            expected_covariance = (
                observation @ state_covariance @ observation.T
                + measurement_covariance
            )
            _covariance_ellipse(
                measurement_axis,
                expected[::-1],
                expected_covariance[::-1, ::-1],
                color=COLORS["measurement"],
                label=r"$H P^{-} H^T + R$",
                alpha=0.22,
            )
            measurement_axis.set_xlim(-2.5, 4.0)
            measurement_axis.set_ylim(-1.5, 5.5)
            _style_axis(
                measurement_axis,
                xlabel="速度读数",
                ylabel="位置读数",
                title="二维测量空间",
            )
            measurement_axis.legend(frameon=False)
        return figure

    def fuse_gaussians_1d(prior_mean, prior_std, measurement_mean, measurement_std):
        prior_variance = prior_std**2
        measurement_variance = measurement_std**2
        gain = prior_variance / (prior_variance + measurement_variance)
        posterior_mean = prior_mean + gain * (measurement_mean - prior_mean)
        posterior_variance = (1.0 - gain) * prior_variance
        return posterior_mean, np.sqrt(posterior_variance), gain

    def plot_gaussian_fusion_1d(
        prior_mean, prior_std, measurement_mean, measurement_std
    ):
        posterior_mean, posterior_std, gain = fuse_gaussians_1d(
            prior_mean, prior_std, measurement_mean, measurement_std
        )
        lower = min(prior_mean - 4 * prior_std, measurement_mean - 4 * measurement_std)
        upper = max(prior_mean + 4 * prior_std, measurement_mean + 4 * measurement_std)
        x_values = np.linspace(lower, upper, 500)
        prior_density = _gaussian_pdf(x_values, prior_mean, prior_std)
        measurement_density = _gaussian_pdf(
            x_values, measurement_mean, measurement_std
        )
        posterior_density = _gaussian_pdf(x_values, posterior_mean, posterior_std)

        figure, axis = plt.subplots(figsize=(9.2, 5.2), constrained_layout=True)
        axis.plot(
            x_values,
            prior_density,
            color=COLORS["prior"],
            linewidth=2.3,
            label="预测分布",
        )
        axis.plot(
            x_values,
            measurement_density,
            color=COLORS["measurement"],
            linewidth=2.3,
            label="测量分布",
        )
        axis.plot(
            x_values,
            posterior_density,
            color=COLORS["posterior"],
            linewidth=3.0,
            label="融合后的后验",
        )
        axis.fill_between(
            x_values,
            posterior_density,
            color=COLORS["posterior"],
            alpha=0.12,
        )
        axis.axvline(posterior_mean, color=COLORS["posterior"], linestyle=":")
        axis.text(
            0.02,
            0.95,
            rf"$K={gain:.3f}$" + "\n" + rf"$\mu^+={posterior_mean:.2f}$"
            + "\n"
            + rf"$\sigma^+={posterior_std:.2f}$",
            transform=axis.transAxes,
            va="top",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "white",
                "edgecolor": COLORS["grid"],
            },
        )
        axis.set_ylim(bottom=0)
        _style_axis(
            axis,
            xlabel="x",
            ylabel="概率密度",
            title="两个一维高斯分布相乘，得到更集中的高斯分布",
        )
        axis.legend(frameon=False, ncol=3)
        return figure

    def plot_gaussian_fusion_2d(separation, prior_rho, measurement_rho):
        prior_mean = np.array([-0.5 * separation, 0.35 * separation])
        measurement_mean = np.array([0.5 * separation, -0.35 * separation])
        prior_covariance = np.array([[2.0, 1.15 * prior_rho], [1.15 * prior_rho, 0.8]])
        measurement_covariance = np.array(
            [[1.0, 0.8 * measurement_rho], [0.8 * measurement_rho, 1.5]]
        )
        prior_precision = np.linalg.inv(prior_covariance)
        measurement_precision = np.linalg.inv(measurement_covariance)
        posterior_covariance = np.linalg.inv(prior_precision + measurement_precision)
        posterior_mean = posterior_covariance @ (
            prior_precision @ prior_mean
            + measurement_precision @ measurement_mean
        )

        figure, axis = plt.subplots(figsize=(8.2, 6.0), constrained_layout=True)
        _covariance_ellipse(
            axis,
            prior_mean,
            prior_covariance,
            color=COLORS["prior"],
            label="预测分布",
            alpha=0.17,
            linestyle="--",
        )
        _covariance_ellipse(
            axis,
            measurement_mean,
            measurement_covariance,
            color=COLORS["measurement"],
            label="测量分布",
            alpha=0.17,
            linestyle="--",
        )
        _covariance_ellipse(
            axis,
            posterior_mean,
            posterior_covariance,
            color=COLORS["posterior"],
            label="乘积 / 后验分布",
            alpha=0.25,
            linewidth=2.6,
        )
        axis.set_xlim(-5.5, 5.5)
        axis.set_ylim(-4.8, 4.8)
        axis.set_aspect("equal", adjustable="box")
        _style_axis(
            axis,
            xlabel=r"$z_1$",
            ylabel=r"$z_2$",
            title="二维高斯融合：后验位于两个估计的重叠区域",
        )
        axis.legend(frameon=False)
        return figure

    def plot_information_flow(stage):
        figure, axis = plt.subplots(figsize=(11.0, 5.2), constrained_layout=True)
        axis.set_xlim(0, 12)
        axis.set_ylim(0, 6)
        axis.axis("off")
        stage_colors = {
            "预测": COLORS["prior"],
            "更新": COLORS["measurement"],
            "完整循环": COLORS["posterior"],
        }
        active = stage_colors[stage]

        boxes = {
            "posterior_old": (
                0.5,
                3.8,
                2.0,
                1.2,
                "上一后验\nx_hat k-1, P k-1",
            ),
            "predict": (3.2, 3.8, 2.0, 1.2, "预测\nF, B, u, Q"),
            "prior": (5.9, 3.8, 2.0, 1.2, "当前先验\nx_hat k, P k"),
            "update": (8.1, 1.2, 2.0, 1.2, "更新\nH, z, R, K"),
            "posterior_new": (5.9, 1.2, 2.0, 1.2, "当前后验\nx_hat k, P k"),
        }

        for name, (x_value, y_value, width, height, text) in boxes.items():
            is_prediction = name in {"posterior_old", "predict", "prior"}
            is_update = name in {"prior", "update", "posterior_new"}
            highlighted = (
                stage == "完整循环"
                or (stage == "预测" and is_prediction)
                or (stage == "更新" and is_update)
            )
            edge_color = active if highlighted else COLORS["grid"]
            face_color = active + "18" if highlighted else COLORS["surface"]
            patch = FancyBboxPatch(
                (x_value, y_value),
                width,
                height,
                boxstyle="round,pad=0.12,rounding_size=0.12",
                linewidth=2.2 if highlighted else 1.4,
                edgecolor=edge_color,
                facecolor=face_color,
            )
            axis.add_patch(patch)
            axis.text(
                x_value + width / 2,
                y_value + height / 2,
                text,
                ha="center",
                va="center",
                fontsize=11,
                color=COLORS["truth"],
            )

        arrows = [
            ((2.5, 4.4), (3.2, 4.4)),
            ((5.2, 4.4), (5.9, 4.4)),
            ((6.9, 3.8), (9.1, 2.4)),
            ((8.1, 1.8), (7.9, 1.8)),
            ((5.9, 1.8), (1.5, 3.8)),
        ]
        for start, end in arrows:
            arrow = FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=15,
                linewidth=2.0,
                color=active,
                connectionstyle="arc3,rad=0.05",
            )
            axis.add_patch(arrow)

        axis.text(
            9.1,
            3.25,
            "传感器信息",
            ha="center",
            color=COLORS["measurement"],
            fontweight="bold",
        )
        axis.text(
            4.2,
            5.45,
            "PREDICT",
            ha="center",
            color=COLORS["prior"],
            fontsize=13,
            fontweight="bold",
        )
        axis.text(
            7.9,
            0.45,
            "UPDATE",
            ha="center",
            color=COLORS["measurement"],
            fontsize=13,
            fontweight="bold",
        )
        axis.set_title(
            f"卡尔曼滤波信息流 · 当前高亮：{stage}",
            loc="left",
            fontweight="bold",
            fontsize=15,
        )
        return figure

    return (
        plot_gaussian_fusion_1d,
        plot_gaussian_fusion_2d,
        plot_information_flow,
        plot_measurement_mapping,
        plot_prediction,
        plot_process_noise,
        plot_robot_demo,
        plot_state_distribution,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 卡尔曼滤波

    卡尔曼滤波器是一个通用且强大的工具，用于在有不确定性的前提下**融合信息**。

    [原文](https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 什么是卡尔曼滤波？
    你可以在任何存在**不确定信息**的动态系统中使用卡尔曼滤波器，，并且你可以对该系统下一步的行为做出**有根据的推测**。

    即便混乱的现实干扰了你推测的理想运动，卡尔曼滤波器通常也能非常出色地推断出实际发生的情况。

    它还能利用那些你可能从未想到要加以利用的、不同现象之间的相关性！

    卡尔曼滤波器非常适用于**持续变化**的系统。其优势在于内存占用低（无需保留除前一状态之外的任何历史数据），且运算速度极快，因此非常适合实时问题和嵌入式系统。

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 卡尔曼滤波器能用来做什么？
    让我们构建一个简单的示例：你制造了一个可以在树林中漫步的小型机器人，该机器人需要精确知晓自身位置以便导航。

    <img src="public/20260619201748.png" width="300" style="display: block; margin: 0 auto;" />

    我们假设机器人的状态为 $\vec{x}_{k}$，仅包含位置和速度：

    $$\vec{x}_{k} = (\vec{p}, \vec{v})$$

    请注意，状态只是描述系统底层配置的一组数值；它可以是任意内容。

    在我们的示例中是位置和速度，但它也可以是油箱中的液体量、汽车引擎的温度、用户在触摸板上的手指位置，或任何你需要跟踪的物理量。

    我们的机器人还配备了一个 GPS **传感器**，精度约为 10 米，这固然不错，但它需要比 10 米更高的定位精度。这片树林中有许多沟壑和悬崖，如果机器人的误差超过几英尺，就可能坠下悬崖。

    因此，仅凭 GPS 是不够的。
    <img src="public/20260619202032.png" width="300" style="display: block; margin: 0 auto;" />
    我们可能还掌握一些关于机器人运动方式的信息：
    1. 它知道发送给轮式电机的指令；
    2. 也知道如果它朝某一方向行进且不受干扰，下一时刻它很可能继续沿同一方向移动。

    但当然，它无法完全掌握自身运动的所有细节：它可能受到风的冲击，车轮可能会轻微打滑，或者在崎岖地形上滚动；因此，车轮转动的量可能并不完全代表机器人实际行驶的距离，基于此进行的**预测**也不会完美。

    GPS **传感器** 告诉我们一些关于状态的信息，但只是间接的，并且带有一定的不确定性或误差。

    我们的 **预测** 告诉我们一些关于机器人如何移动的信息，但同样只是间接的，并且带有一定的不确定性或误差。

    但是，如果我们利用所有可用的信息，能否获得比**任一单独估计**更优的结果？

    答案当然是肯定的，而这正是卡尔曼滤波器的用途所在。

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 卡尔曼滤波器如何看待你的问题

    回到上一节中提出的问题。我们将继续使用仅包含位置和速度的简单状态；为了说明的简单性，我们分析一个仅有一维运动的机器人。

    $$\vec{x} = \begin{bmatrix} p \\ v \end{bmatrix}$$

    我们并不知道**实际**机器人的位置和速度是多少（Ground Truth）；存在一系列可能的位置（$p$）与速度（$v$）的组合，但其中某些组合比其他组合更可能发生。

    卡尔曼滤波器假设这两个变量（在我们的例子中是位置和速度）均为随机变量，且服从**高斯分布**。

    下面的动画让你可以设置这个一维运动的机器人的初速度和加速度，观察理想情况下它的状态。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.iframe(
    r"""
    <!doctype html>
    <html>
    <head>
    <meta charset="utf-8" />
    <style>
      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: transparent;
      }

      .card {
        width: 840px;
        padding: 22px;
        border-radius: 24px;
        background: linear-gradient(135deg, #f8fbff, #ffffff);
        border: 1px solid #dbeafe;
        box-shadow: 0 18px 45px rgba(30, 64, 175, 0.12);
      }

      .top {
        display: flex;
        justify-content: space-between;
        gap: 18px;
        align-items: flex-start;
        margin-bottom: 18px;
      }

      h2 {
        margin: 0;
        color: #0f172a;
        font-size: 24px;
      }

      .subtitle {
        margin-top: 6px;
        color: #475569;
        font-size: 14px;
      }

      .badge {
        padding: 8px 12px;
        border-radius: 999px;
        background: #e0f2fe;
        color: #0369a1;
        font-weight: 800;
        font-size: 14px;
        white-space: nowrap;
      }

      .controls {
        display: grid;
        grid-template-columns: 1fr 1fr auto auto;
        gap: 14px;
        align-items: end;
        padding: 14px;
        border-radius: 18px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        margin-bottom: 16px;
      }

      .control label {
        display: block;
        font-size: 13px;
        font-weight: 800;
        color: #334155;
        margin-bottom: 6px;
      }

      input[type="range"] {
        width: 100%;
      }

      .readout {
        margin-top: 4px;
        font-size: 13px;
        color: #0f172a;
        font-weight: 700;
      }

      button {
        height: 42px;
        border: 0;
        border-radius: 999px;
        padding: 0 18px;
        font-weight: 900;
        cursor: pointer;
        transition: transform 0.12s ease, box-shadow 0.12s ease;
      }

      button:hover {
        transform: translateY(-1px);
      }

      #play {
        color: white;
        background: linear-gradient(135deg, #22c55e, #16a34a);
        box-shadow: 0 10px 20px rgba(22, 163, 74, 0.25);
      }

      #reset {
        color: #334155;
        background: #e2e8f0;
      }

      .scene {
        position: relative;
        width: 796px;
        height: 300px;
        overflow: hidden;
        border-radius: 20px;
        border: 1px solid #bfdbfe;
        background:
          radial-gradient(circle at 14% 20%, rgba(253, 230, 138, 0.9), transparent 10%),
          radial-gradient(circle at 22% 18%, rgba(125, 211, 252, 0.35), transparent 28%),
          linear-gradient(#eff6ff 0%, #f8fafc 64%, #e2e8f0 64%, #cbd5e1 100%);
      }

      .direction {
        position: absolute;
        right: 36px;
        top: 30px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.82);
        color: #334155;
        font-weight: 800;
      }

      .finish-line {
        position: absolute;
        left: 694px;
        bottom: 78px;
        width: 5px;
        height: 150px;
        border-radius: 999px;
        background: repeating-linear-gradient(
          to bottom,
          #111827 0px,
          #111827 10px,
          #ffffff 10px,
          #ffffff 20px
        );
      }

      .flag {
        position: absolute;
        top: -30px;
        left: -10px;
        font-size: 28px;
      }

      .finish-label {
        position: absolute;
        left: -38px;
        bottom: -34px;
        width: 90px;
        text-align: center;
        color: #334155;
        font-size: 13px;
        font-weight: 800;
      }

      .track {
        position: absolute;
        left: 70px;
        right: 70px;
        bottom: 58px;
      }

      .track-line {
        height: 9px;
        border-radius: 999px;
        background: linear-gradient(90deg, #334155, #64748b);
        box-shadow: 0 5px 12px rgba(15, 23, 42, 0.25);
      }

      .ticks {
        display: flex;
        justify-content: space-between;
        margin-top: 8px;
        color: #475569;
        font-size: 13px;
        font-weight: 700;
      }

      .robot {
        position: absolute;
        left: 70px;
        bottom: 76px;
        width: 96px;
        height: 124px;
        transform: translateX(0px);
        will-change: transform;
      }

      .antenna {
        position: absolute;
        left: 46px;
        top: 0;
        width: 4px;
        height: 24px;
        background: #334155;
        border-radius: 999px;
      }

      .antenna-dot {
        position: absolute;
        left: -6px;
        top: -9px;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #38bdf8;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.9);
      }

      .head {
        position: absolute;
        left: 20px;
        top: 21px;
        width: 58px;
        height: 42px;
        border-radius: 16px;
        background: linear-gradient(145deg, #e2e8f0, #ffffff);
        border: 3px solid #334155;
        box-shadow: inset 0 0 0 2px rgba(255, 255, 255, 0.7);
      }

      .face-screen {
        position: absolute;
        left: 8px;
        top: 8px;
        width: 42px;
        height: 24px;
        border-radius: 9px;
        background: #0f172a;
      }

      .eye {
        position: absolute;
        top: 7px;
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #67e8f9;
      }

      .left-eye {
        left: 9px;
      }

      .right-eye {
        right: 9px;
      }

      .smile {
        position: absolute;
        left: 15px;
        bottom: 5px;
        width: 13px;
        height: 6px;
        border-bottom: 2px solid #67e8f9;
        border-radius: 0 0 999px 999px;
      }

      .neck {
        position: absolute;
        left: 42px;
        top: 63px;
        width: 14px;
        height: 10px;
        background: #64748b;
      }

      .body {
        position: absolute;
        left: 12px;
        top: 72px;
        width: 72px;
        height: 42px;
        border-radius: 18px 18px 12px 12px;
        background: linear-gradient(145deg, #60a5fa, #2563eb);
        border: 3px solid #1e3a8a;
        box-shadow: inset 0 3px 8px rgba(255, 255, 255, 0.45);
      }

      .panel {
        position: absolute;
        left: 18px;
        top: 10px;
        display: flex;
        gap: 6px;
      }

      .light {
        width: 8px;
        height: 8px;
        border-radius: 50%;
      }

      .light-1 { background: #22c55e; }
      .light-2 { background: #facc15; }
      .light-3 { background: #fb7185; }

      .arm {
        position: absolute;
        top: 12px;
        width: 18px;
        height: 7px;
        background: #334155;
        border-radius: 999px;
      }

      .left-arm {
        left: -16px;
        transform: rotate(18deg);
      }

      .right-arm {
        right: -16px;
        transform: rotate(-18deg);
      }

      .wheel {
        position: absolute;
        bottom: -6px;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background:
          radial-gradient(circle, #f8fafc 0 24%, transparent 25%),
          conic-gradient(#0f172a 0 25%, #94a3b8 25% 50%, #0f172a 50% 75%, #94a3b8 75% 100%);
        border: 4px solid #111827;
      }

      .left-wheel { left: 15px; }
      .right-wheel { right: 15px; }

      .hub {
        position: absolute;
        left: 8px;
        top: 8px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #e2e8f0;
      }

      .info-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-top: 14px;
      }

      .info-box {
        padding: 12px;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
      }

      .info-label {
        color: #64748b;
        font-size: 12px;
        margin-bottom: 4px;
      }

      .info-value {
        color: #0f172a;
        font-size: 15px;
        font-weight: 900;
      }

      .warning {
        margin-top: 10px;
        min-height: 20px;
        color: #dc2626;
        font-weight: 800;
        font-size: 14px;
      }
    </style>
    </head>

    <body>
      <div class="card">
        <div class="top">
          <div>
            <h2>一维匀加速机器人</h2>
            <div class="subtitle">
              固定路程：从 p = 0 m 运动到 p = 10 m。状态只包含位置和速度：state = [p, v]
            </div>
          </div>
          <div class="badge">只能向右平移</div>
        </div>

        <div class="controls">
          <div class="control">
            <label for="v0">初速度 v₀</label>
            <input id="v0" type="range" min="0" max="5" step="0.1" value="1.0" />
            <div class="readout"><span id="v0Text">1.0</span> m/s</div>
          </div>

          <div class="control">
            <label for="acc">加速度 a</label>
            <input id="acc" type="range" min="0" max="3" step="0.1" value="0.5" />
            <div class="readout"><span id="accText">0.5</span> m/s²</div>
          </div>

          <button id="play">▶ 播放</button>
          <button id="reset">↺ 复位</button>
        </div>

        <div class="scene">
          <div class="direction">→</div>

          <div class="finish-line">
            <div class="flag">🏁</div>
            <div class="finish-label">p = 10 m</div>
          </div>

          <div class="track">
            <div class="track-line"></div>
            <div class="ticks">
              <span>0 m</span>
              <span>5 m</span>
              <span>10 m</span>
            </div>
          </div>

          <div class="robot" id="robot">
            <div class="antenna">
              <div class="antenna-dot"></div>
            </div>

            <div class="head">
              <div class="face-screen">
                <div class="eye left-eye"></div>
                <div class="eye right-eye"></div>
                <div class="smile"></div>
              </div>
            </div>

            <div class="neck"></div>

            <div class="body">
              <div class="panel">
                <div class="light light-1"></div>
                <div class="light light-2"></div>
                <div class="light light-3"></div>
              </div>
              <div class="arm left-arm"></div>
              <div class="arm right-arm"></div>
            </div>

            <div class="wheel left-wheel" id="wheel1">
              <div class="hub"></div>
            </div>
            <div class="wheel right-wheel" id="wheel2">
              <div class="hub"></div>
            </div>
          </div>
        </div>

        <div class="info-grid">
          <div class="info-box">
            <div class="info-label">当前位置</div>
            <div class="info-value" id="xInfo">p = 0.00 m</div>
          </div>
          <div class="info-box">
            <div class="info-label">当前速度</div>
            <div class="info-value" id="vInfo">v = 0.00 m/s</div>
          </div>
          <div class="info-box">
            <div class="info-label">物理时间</div>
            <div class="info-value" id="tInfo">t = 0.00 s</div>
          </div>
          <div class="info-box">
            <div class="info-label">终点速度</div>
            <div class="info-value" id="vendInfo">v = -- m/s</div>
          </div>
        </div>

        <div class="warning" id="warning"></div>
      </div>

    <script>
    (() => {
      const S = 10.0;
      const travelPx = 600;

      const v0Slider = document.getElementById("v0");
      const accSlider = document.getElementById("acc");
      const v0Text = document.getElementById("v0Text");
      const accText = document.getElementById("accText");

      const playButton = document.getElementById("play");
      const resetButton = document.getElementById("reset");

      const robot = document.getElementById("robot");
      const wheel1 = document.getElementById("wheel1");
      const wheel2 = document.getElementById("wheel2");

      const xInfo = document.getElementById("xInfo");
      const vInfo = document.getElementById("vInfo");
      const tInfo = document.getElementById("tInfo");
      const vendInfo = document.getElementById("vendInfo");
      const warning = document.getElementById("warning");

      let animationId = null;
      let startTime = null;

      function updateSliderText() {
        v0Text.textContent = Number(v0Slider.value).toFixed(1);
        accText.textContent = Number(accSlider.value).toFixed(1);
      }

      function resetRobot() {
        if (animationId !== null) {
          cancelAnimationFrame(animationId);
          animationId = null;
        }
        startTime = null;
        robot.style.transform = "translateX(0px)";
        wheel1.style.transform = "rotate(0deg)";
        wheel2.style.transform = "rotate(0deg)";
        xInfo.textContent = "x = 0.00 m";
        vInfo.textContent = "v = 0.00 m/s";
        tInfo.textContent = "t = 0.00 s";
        warning.textContent = "";
      }

      function solveTime(v0, a) {
        if (v0 === 0 && a === 0) {
          return null;
        }

        if (Math.abs(a) < 1e-12) {
          return S / v0;
        }

        return (-v0 + Math.sqrt(v0 * v0 + 2 * a * S)) / a;
      }

      function play() {
        resetRobot();

        const v0 = Number(v0Slider.value);
        const a = Number(accSlider.value);

        const tEnd = solveTime(v0, a);

        if (tEnd === null || !Number.isFinite(tEnd)) {
          warning.textContent = "初速度和加速度不能同时为 0，否则机器人不会运动。";
          vendInfo.textContent = "v = -- m/s";
          return;
        }

        const vEnd = v0 + a * tEnd;
        vendInfo.textContent = `v = ${vEnd.toFixed(2)} m/s`;

        // 物理时间仍然按 tEnd 计算；为了观看方便，把显示时间压缩到 1.5~8 秒
        const displayDuration = Math.max(1.5, Math.min(tEnd, 8.0));
        const wheelSpinFactor = 8.0;

        function step(timestamp) {
          if (startTime === null) {
            startTime = timestamp;
          }

          const elapsed = (timestamp - startTime) / 1000;
          const progress = Math.min(elapsed / displayDuration, 1.0);

          const t = progress * tEnd;
          const x = Math.min(v0 * t + 0.5 * a * t * t, S);
          const v = v0 + a * t;

          const px = travelPx * x / S;
          const wheelAngle = px * wheelSpinFactor;

          robot.style.transform = `translateX(${px}px)`;
          wheel1.style.transform = `rotate(${wheelAngle}deg)`;
          wheel2.style.transform = `rotate(${wheelAngle}deg)`;

          xInfo.textContent = `x = ${x.toFixed(2)} m`;
          vInfo.textContent = `v = ${v.toFixed(2)} m/s`;
          tInfo.textContent = `t = ${t.toFixed(2)} s`;

          if (progress < 1.0) {
            animationId = requestAnimationFrame(step);
          } else {
            animationId = null;
          }
        }

        animationId = requestAnimationFrame(step);
      }

      v0Slider.addEventListener("input", updateSliderText);
      accSlider.addEventListener("input", updateSliderText);
      playButton.addEventListener("click", play);
      resetButton.addEventListener("click", resetRobot);

      updateSliderText();
      resetRobot();
    })();
    </script>
    </body>
    </html>
    """,
    width="100%",
    height="650px",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    卡尔曼滤波器假设状态包含的两个变量（在我们的例子中是位置和速度）均为随机变量，且服从**高斯分布**。也就是:
    $$p\sim\mathcal{N}(\mu_{\text{position}},\sigma^{2}_{{\text{position}}})$$
    $$v\sim\mathcal{N}(\mu_{\text{velocity}},\sigma^{2}_{{\text{velocity}}})$$
    每个变量都有一个**均值** $\mu$，它是随机分布的中心（即最可能的状态），以及一个**方差** $\sigma^2$，表示不确定性。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mu_p = mo.ui.slider(
        start=-10,
        stop=10,
        step=0.1,
        value=0,
        label=r"位置均值 $\mu_p$",
    )

    sigma_p = mo.ui.slider(
        start=0.2,
        stop=6,
        step=0.1,
        value=2,
        label=r"位置标准差 $\sigma_p$",
    )

    mu_v = mo.ui.slider(
        start=-10,
        stop=10,
        step=0.1,
        value=0,
        label=r"速度均值 $\mu_v$",
    )

    sigma_v = mo.ui.slider(
        start=0.2,
        stop=6,
        step=0.1,
        value=2,
        label=r"速度标准差 $\sigma_v$",
    )

    mo.vstack(
        [
            mo.md('''
            如果位置和速度二者是独立的两个变量，有：
                \[\mathcal{G}(p, v) = \mathcal{G}(p) \mathcal{G}(v)\]
            那么我们可以通过下面的滑块控制均值和**标准差**来调整位置和速度的高斯分布，以及他们的联合概率分布。
            '''),
            mo.hstack([mu_p, sigma_p]),
            mo.hstack([mu_v, sigma_v]),

        ]
    )
    return mu_p, mu_v, sigma_p, sigma_v


@app.cell(hide_code=True)
def _(mo, mu_p, mu_v, sigma_p, sigma_v):
    mo.md(rf"""
    当前参数：$\quad$ $p \sim \mathcal{{N}}({mu_p.value:.2f}, {sigma_p.value ** 2:.2f})$ $\quad$ $v \sim \mathcal{{N}}({mu_v.value:.2f}, {sigma_v.value ** 2:.2f})$
    """)
    return


@app.cell(hide_code=True)
def _(GridSpec, figure_as_svg, mu_p, mu_v, np, plt, sigma_p, sigma_v):
    def gaussian_1d(x, mu, sigma):
        return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )


    p_min = min(-12, mu_p.value - 4 * sigma_p.value)
    p_max = max(12, mu_p.value + 4 * sigma_p.value)

    v_min = min(-12, mu_v.value - 4 * sigma_v.value)
    v_max = max(12, mu_v.value + 4 * sigma_v.value)

    p = np.linspace(p_min, p_max, 300)
    v = np.linspace(v_min, v_max, 300)

    P, V = np.meshgrid(p, v)

    pdf_p = gaussian_1d(p, mu_p.value, sigma_p.value)
    pdf_v = gaussian_1d(v, mu_v.value, sigma_v.value)

    joint_pdf = gaussian_1d(P, mu_p.value, sigma_p.value) * gaussian_1d(
        V, mu_v.value, sigma_v.value
    )


    fig = plt.figure(figsize=(9, 8))

    gs = GridSpec(
        2,
        2,
        width_ratios=[4, 1.2],
        height_ratios=[1.2, 4],
        hspace=0.2,
        wspace=0.2,
    )

    ax_top = fig.add_subplot(gs[0, 0])
    ax_joint = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax_joint)


    # 上方：位置的一维高斯分布
    ax_top.plot(p, pdf_p, linewidth=2)
    ax_top.fill_between(p, pdf_p, alpha=0.25)
    ax_top.set_xlim(p_min, p_max)
    ax_top.set_ylabel(r"$f(p)$")
    ax_top.set_title(
        f"位置分布：μp={mu_p.value:.1f}, σp={sigma_p.value:.1f}"
    )
    ax_top.tick_params(axis="x", labelbottom=False)


    # 中间：二维联合概率分布
    heatmap = ax_joint.imshow(
        joint_pdf,
        extent=[p_min, p_max, v_min, v_max],
        origin="lower",
        aspect="auto",
        cmap="viridis",
    )

    ax_joint.contour(
        P,
        V,
        joint_pdf,
        colors="white",
        alpha=0.5,
        linewidths=0.8,
    )

    ax_joint.axvline(mu_p.value, color="white", linestyle="--", linewidth=1)
    ax_joint.axhline(mu_v.value, color="white", linestyle="--", linewidth=1)

    ax_joint.set_xlabel("位置 p")
    ax_joint.set_ylabel("速度 v")
    ax_joint.set_title("联合概率密度 f(p, v)")

    cbar = fig.colorbar(heatmap, ax=ax_joint, fraction=0.046, pad=0.04)
    cbar.set_label("概率密度")


    # 右侧：速度的一维高斯分布
    ax_right.plot(pdf_v, v, linewidth=2)
    ax_right.fill_betweenx(v, pdf_v, alpha=0.25)
    ax_right.set_xlabel(r"$f(v)$")
    ax_right.set_title(
        f"速度分布：μv={mu_v.value:.1f}, σv={sigma_v.value:.1f}"
    )
    ax_right.tick_params(axis="y", labelleft=False)


    fig.suptitle("位置-速度联合高斯分布可视化", fontsize=15, y=0.98)
    figure_as_svg(fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    可以观察到当代表不确定性的方差越小时，图像中的亮点越集中。而亮斑处的状态（也就是位置与速度的组合），比其他地方的组合更加可能发生。

    在上图中，位置和速度是**不相关的**，这意味着一个变量的状态无法提供关于另一个变量的任何信息。

    下面我们来看更加贴合实际的情况：位置和速度是**相关的**。观察到某一特定位置的可能性取决于你所具有的速度。因为当我们基于旧位置估计新位置时，如果速度很高，我们可能在同样的时间内移动得更多，因此位置会更远。如果移动缓慢，则不会走那么远。

    这种关系非常值得持续跟踪，因为它为我们提供了**更多信息**：一次测量可以告诉我们其他变量可能是什么样的。

    而这正是卡尔曼滤波器的目标——我们希望从不确定的测量中尽可能多地榨取信息！

    这种相关性由**协方差矩阵**来捕捉。简而言之，矩阵的每个元素 $\Sigma_{ij}$ 表示第 $i$ 个状态变量与第 $j$ 个状态变量之间的相关程度。（你或许可以猜到，协方差矩阵是**对称的**，这意味着交换 $i$ 和 $j$ 没有影响）。协方差矩阵通常记为 "$\mathbf{\Sigma}$"，因此我们称其第$i$行、第$j$列的元素为 "$\Sigma_{ij}$"。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    Sigma_pp_slider_cov = mo.ui.slider(
        start=0.2,
        stop=36,
        step=0.1,
        value=4.0,
        label=r"$\Sigma_{pp}$：位置方差",
    )

    Sigma_pv_slider_cov = mo.ui.slider(
        start=-20,
        stop=20,
        step=0.1,
        value=0.0,
        label=r"$\Sigma_{pv}=\Sigma_{vp}$：位置-速度协方差",
    )

    Sigma_vv_slider_cov = mo.ui.slider(
        start=0.2,
        stop=36,
        step=0.1,
        value=4.0,
        label=r"$\Sigma_{vv}$：速度方差",
    )


    mo.vstack(
        [
            mo.md("**通过滑块调整协方差矩阵的三个元素，如果形成的协方差矩阵不满足半正定性，会产生提醒。**"),
            Sigma_pp_slider_cov,
            Sigma_pv_slider_cov,
            Sigma_vv_slider_cov,
        ]
    )
    return Sigma_pp_slider_cov, Sigma_pv_slider_cov, Sigma_vv_slider_cov


@app.cell(hide_code=True)
def _(
    Sigma_pp_slider_cov,
    Sigma_pv_slider_cov,
    Sigma_vv_slider_cov,
    figure_as_svg,
    mo,
    mu_p,
    mu_v,
    np,
    plt,
):
    def bivariate_gaussian_cov(pos_cov, mean_cov, Sigma_cov):
        inv_Sigma_cov = np.linalg.inv(Sigma_cov)
        det_Sigma_cov = np.linalg.det(Sigma_cov)

        diff_cov = pos_cov - mean_cov

        exponent_cov = np.einsum(
            "...i,ij,...j->...",
            diff_cov,
            inv_Sigma_cov,
            diff_cov,
        )

        return 1 / (2 * np.pi * np.sqrt(det_Sigma_cov)) * np.exp(
            -0.5 * exponent_cov
        )


    # 读取均值滑块
    mu_p_value_cov = mu_p.value
    mu_v_value_cov = mu_v.value

    # 直接读取协方差矩阵三个独立元素
    Sigma_pp_value_cov = Sigma_pp_slider_cov.value
    Sigma_pv_value_cov = Sigma_pv_slider_cov.value
    Sigma_vv_value_cov = Sigma_vv_slider_cov.value

    Sigma_vp_value_cov = Sigma_pv_value_cov

    Sigma_cov = np.array(
        [
            [Sigma_pp_value_cov, Sigma_pv_value_cov],
            [Sigma_vp_value_cov, Sigma_vv_value_cov],
        ]
    )

    mean_cov = np.array([mu_p_value_cov, mu_v_value_cov])

    det_Sigma_cov = np.linalg.det(Sigma_cov)
    is_valid_cov = (
        Sigma_pp_value_cov > 0
        and Sigma_vv_value_cov > 0
        and det_Sigma_cov > 0
    )

    matrix_md_cov = mo.md(
        rf"""
        ## 协方差矩阵

        \[
        \mathbf{{\Sigma}}
        =
        \begin{{bmatrix}}
        \Sigma_{{pp}} & \Sigma_{{pv}} \\
        \Sigma_{{vp}} & \Sigma_{{vv}}
        \end{{bmatrix}}
        =
        \begin{{bmatrix}}
        {Sigma_pp_value_cov:.3f} & {Sigma_pv_value_cov:.3f} \\
        {Sigma_vp_value_cov:.3f} & {Sigma_vv_value_cov:.3f}
        \end{{bmatrix}}
        \]

        其中：

        \[
        \Sigma_{{pv}} = \Sigma_{{vp}} = {Sigma_pv_value_cov:.3f}
        \]

        行列式：

        \[
        \det(\mathbf{{\Sigma}})
        =
        \Sigma_{{pp}}\Sigma_{{vv}} - \Sigma_{{pv}}^2
        =
        {det_Sigma_cov:.3f}
        \]
        """
    )


    if not is_valid_cov:
        output_cov = mo.vstack(
            [
                matrix_md_cov,
                mo.md(
                    r"""
                    ⚠️ 当前协方差矩阵不是合法的协方差矩阵，因此不能生成二维高斯分布。

                    合法条件是：

                    \[
                    \Sigma_{pp} > 0
                    \]

                    \[
                    \Sigma_{vv} > 0
                    \]

                    \[
                    \Sigma_{pp}\Sigma_{vv} - \Sigma_{pv}^2 > 0
                    \]

                    也就是说，协方差 \(\Sigma_{pv}\) 的绝对值不能太大。
                    """
                ),
            ]
        )

    else:
        sigma_p_from_matrix_cov = np.sqrt(Sigma_pp_value_cov)
        sigma_v_from_matrix_cov = np.sqrt(Sigma_vv_value_cov)

        # 画图范围
        p_min_cov = mu_p_value_cov - 4 * sigma_p_from_matrix_cov
        p_max_cov = mu_p_value_cov + 4 * sigma_p_from_matrix_cov

        v_min_cov = mu_v_value_cov - 4 * sigma_v_from_matrix_cov
        v_max_cov = mu_v_value_cov + 4 * sigma_v_from_matrix_cov

        p_cov = np.linspace(p_min_cov, p_max_cov, 350)
        v_cov = np.linspace(v_min_cov, v_max_cov, 350)

        P_cov, V_cov = np.meshgrid(p_cov, v_cov)
        pos_cov = np.dstack((P_cov, V_cov))

        joint_pdf_cov = bivariate_gaussian_cov(pos_cov, mean_cov, Sigma_cov)

        fig_cov, ax_joint_cov = plt.subplots(figsize=(8, 7))

        heatmap_cov = ax_joint_cov.imshow(
            joint_pdf_cov,
            extent=[p_min_cov, p_max_cov, v_min_cov, v_max_cov],
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )

        ax_joint_cov.contour(
            P_cov,
            V_cov,
            joint_pdf_cov,
            colors="white",
            alpha=0.6,
            linewidths=1.0,
        )

        ax_joint_cov.axvline(
            mu_p_value_cov,
            color="white",
            linestyle="--",
            linewidth=1,
        )

        ax_joint_cov.axhline(
            mu_v_value_cov,
            color="white",
            linestyle="--",
            linewidth=1,
        )

        ax_joint_cov.set_xlabel("位置 p")
        ax_joint_cov.set_ylabel("速度 v")
        ax_joint_cov.set_title("由协方差矩阵决定的二维高斯分布 f(p, v)")

        cbar_cov = fig_cov.colorbar(
            heatmap_cov,
            ax=ax_joint_cov,
            fraction=0.046,
            pad=0.04,
        )
        cbar_cov.set_label("概率密度")

        fig_cov.tight_layout()

        output_cov = mo.vstack(
            [
                matrix_md_cov,
                figure_as_svg(fig_cov),
                mo.md(
                    rf"""
                    当前矩阵是合法协方差矩阵。
                    """
                ),
            ]
        )
    #                 # 对应的相关系数为：

                    # \[
                    # \rho
                    # =
                    # \frac{{\Sigma_{{pv}}}}{{\sqrt{{\Sigma_{{pp}}\Sigma_{{vv}}}}}}
                    # =
                    # {Sigma_pv_value_cov / np.sqrt(Sigma_pp_value_cov * Sigma_vv_value_cov):.3f}
                    # \]

                    # 这里的 \(\rho\) 不是滑块控制量，而是由协方差矩阵自动计算出来的结果。

    output_cov
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 用矩阵描述问题
    我们将关于状态的知识建模为一个高斯分布，因此在时刻 $k$ 需要两条信息：
    1. 最佳估计记为 $\mathbf{\hat{x}_k}$（即均值，在其他地方记为 $\mu$）；
    2. 及其协方差矩阵 $\mathbf{P_k}$。

    $$
    \begin{aligned}
    \mathbf{\hat{x}}_k &= \begin{bmatrix}
    \text{position}\\
    \text{velocity}
    \end{bmatrix}\\
    \mathbf{P}_k &=
    \begin{bmatrix}
    \Sigma_{pp} & \Sigma_{pv} \\
    \Sigma_{vp} & \Sigma_{vv} \\
    \end{bmatrix}
    \end{aligned}$$

    （当然，我们这里仅使用位置和速度，但值得记住的是，状态可以包含任意数量的变量，并代表任何你想要描述的物理量）。

    接下来，我们需要某种方法，根据**时刻 k-1 的当前状态**来**预测时刻 k 的下一状态**。请记住，我们并不知道哪个状态是"真实"的，但我们的预测函数并不在意。它只是对*所有可能的状态*进行操作，并给出新的分布。

    我们可以用一个矩阵 $\mathbf{F_k}$ 来表示这一预测步骤：

    $\mathbf{F_k}$将原始估计中的*每一个点*移动到新的预测位置，即如果原始估计正确，系统将移动到的位置。
    """)
    return


@app.cell(hide_code=True)
def _(FancyArrowPatch, figure_as_svg, np, pe, plt):
    def _():
        # -------- 坐标系：和参考图 2 保持一致 --------
        p_min, p_max = -8, 8
        v_min, v_max = -15, 1

        p = np.linspace(p_min, p_max, 600)
        v = np.linspace(v_min, v_max, 600)
        P_grid, V_grid = np.meshgrid(p, v)

        grid = np.stack([P_grid, V_grid], axis=-1)


        def gaussian_pdf_2d(grid, mu, cov):
            diff = grid - mu
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)

            expo = -0.5 * np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
            norm = 1.0 / (2 * np.pi * np.sqrt(det_cov))

            return norm * np.exp(expo)


        # -------- 写死的示意数值 --------
        # 时刻 k-1 的估计
        mu_prev = np.array([-4.0, -10.5])

        P_prev = np.array([
            [1.50, 0.75],
            [0.75, 1.25],
        ])

        # 状态转移矩阵 F_k
        F_k = np.array([
            [0.70, -0.33],
            [-0.20, 0.38],
        ])

        # 预测到时刻 k
        mu_pred = F_k @ mu_prev
        P_pred = F_k @ P_prev @ F_k.T


        # -------- 计算两个高斯分布 --------
        Z_prev = gaussian_pdf_2d(grid, mu_prev, P_prev)
        Z_pred = gaussian_pdf_2d(grid, mu_pred, P_pred)

        # 用相对密度做示意图，便于两个分布同时看清楚
        Z_prev_rel = Z_prev / Z_prev.max()
        Z_pred_rel = Z_pred / Z_pred.max()

        Z = 0.70 * Z_prev_rel + 1.00 * Z_pred_rel


        # -------- 绘图 --------
        fig, ax = plt.subplots(figsize=(10, 8), dpi=160)

        im = ax.imshow(
            Z,
            extent=[p_min, p_max, v_min, v_max],
            origin="lower",
            cmap="viridis",
            aspect="auto",
            vmin=0,
            vmax=1.05,
        )

        # 等高线
        levels = [0.12, 0.28, 0.44, 0.60, 0.76]

        ax.contour(
            P_grid,
            V_grid,
            Z_prev_rel,
            levels=levels,
            colors="white",
            linewidths=1.4,
            alpha=0.35,
        )

        ax.contour(
            P_grid,
            V_grid,
            Z_pred_rel,
            levels=levels,
            colors="white",
            linewidths=1.7,
            alpha=0.65,
        )

        # 参考图 2 中的虚线坐标
        ax.axvline(0, linestyle="--", linewidth=1.4, color="white", alpha=0.75)
        ax.axhline(-7, linestyle="--", linewidth=1.4, color="white", alpha=0.75)

        text_effect = [
            pe.withStroke(linewidth=4, foreground="black", alpha=0.55)
        ]

        # 均值点
        ax.scatter(
            *mu_prev,
            s=42,
            color="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )

        ax.scatter(
            *mu_pred,
            s=50,
            color="white",
            edgecolor="black",
            linewidth=0.8,
            zorder=5,
        )


        # -------- 画出“每一个点都会被 F_k 移动”的示意箭头 --------
        offsets = np.array([
            [0.00, 0.00],
            [0.85, 0.15],
            [-0.65, 0.30],
            [0.25, -0.75],
            [-0.35, -0.55],
        ])

        L = np.linalg.cholesky(P_prev)

        pts_prev = mu_prev + offsets @ L.T
        pts_pred = (F_k @ pts_prev.T).T

        for start, end in zip(pts_prev, pts_pred):
            ax.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="->",
                    mutation_scale=12,
                    linewidth=1.15,
                    color="white",
                    alpha=0.38,
                    connectionstyle="arc3,rad=0.05",
                    zorder=4,
                )
            )

        ax.scatter(
            pts_prev[:, 0],
            pts_prev[:, 1],
            s=18,
            color="white",
            alpha=0.70,
            zorder=5,
        )

        ax.scatter(
            pts_pred[:, 0],
            pts_pred[:, 1],
            s=18,
            color="white",
            alpha=0.85,
            zorder=5,
        )


        # 主箭头：分布整体从 k-1 预测到 k
        ax.add_patch(
            FancyArrowPatch(
                mu_prev + np.array([0.25, 0.25]),
                mu_pred + np.array([-0.15, -0.15]),
                arrowstyle="->",
                mutation_scale=28,
                linewidth=3.0,
                color="white",
                alpha=0.95,
                connectionstyle="arc3,rad=0.22",
                zorder=6,
            )
        )

        mid = (mu_prev + mu_pred) / 2

        ax.text(
            mid[0] + 0.15,
            mid[1] + 0.75,
            r"$\mathbf{F}_k$",
            fontsize=24,
            color="white",
            path_effects=text_effect,
            zorder=7,
        )


        # -------- 文字标注 --------
        ax.text(
            mu_prev[0] - 2.7,
            mu_prev[1] - 1.7,
            "时刻 k-1\nx_hat k-1, P k-1",
            fontsize=14,
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.45, edgecolor="none"),
            zorder=7,
        )

        ax.text(
            mu_pred[0] + 0.6,
            mu_pred[1] + 0.7,
            "预测到时刻 k\nx_hat k, P k",
            fontsize=14,
            color="white",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.45, edgecolor="none"),
            zorder=7,
        )

        ax.text(
            3.35,
            -13.4,
            r"$\hat{\mathbf{x}}_k=\mathbf{F}_k\hat{\mathbf{x}}_{k-1}$" "\n"
            r"$\mathbf{P}_k=\mathbf{F}_k\mathbf{P}_{k-1}\mathbf{F}_k^\mathsf{T}$",
            fontsize=13,
            color="white",
            path_effects=text_effect,
            zorder=7,
        )


        # -------- 坐标轴：和参考图 2 一致 --------
        ax.set_xlim(p_min, p_max)
        ax.set_ylim(v_min, v_max)

        ax.set_xticks(np.arange(-8, 8, 2))
        ax.set_yticks(np.arange(0, -16, -2))

        ax.set_xlabel("位置 p", fontsize=13)
        ax.set_ylabel("速度 v", fontsize=13)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("概率密度（示意）", rotation=90, labelpad=15, fontsize=12)

        fig.tight_layout()
        return fig


    figure_as_svg(_())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    回到一维机器人的例子中来，如何使用矩阵来预测未来下一时刻的位置和速度？我们将使用一个非常基本的运动学公式：

    $${\color{deeppink}{p_k}} = {\color{royalblue}p_{k-1}} + {\color{royalblue}\Delta t {v_{k-1}}} \\
    {\color{deeppink}{v_k}} = \color{royalblue}{v_{k-1}}\tag{1}$$

    写成矩阵形式为：

    $$\begin{align}
    \color{deeppink}{\mathbf{\hat{x}}_k} &= \begin{bmatrix}
    1 & \Delta t \\
    0 & 1
    \end{bmatrix} \color{royalblue}{\mathbf{\hat{x}}_{k-1}} \tag{2}\\
    &= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}} \tag{3}
    \end{align}$$

    我们现在拥有了一个预测出下一状态的**预测矩阵**，但我们仍然不知道如何更新协方差矩阵。

    这正是我们需要另一个公式的地方。如果我们将分布中的每个点乘以一个矩阵 $\color{firebrick}{\mathbf{A}}$，其协方差矩阵 $\Sigma$ 会发生什么变化？

    这里的结论是：

    $$\begin{equation}
    \begin{split}
    Cov(x) &= \Sigma\\
    Cov(\color{firebrick}{\mathbf{A}}x) &= \color{firebrick}{\mathbf{A}} \Sigma \color{firebrick}{\mathbf{A}}^T
    \end{split}
    \end{equation}\tag{4}$$

    过程推导见于附录2。

    因此，将方程(3)与方程(4)结合，可以得到：

    $$\begin{equation}
    \begin{split}
    \color{deeppink}{\mathbf{\hat{x}}_k} &= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}} \\
    \color{deeppink}{\mathbf{P}_k} &= \mathbf{F_k} \color{royalblue}{\mathbf{P}_{k-1}} \mathbf{F}_k^T
    \end{split}
    \end{equation}\tag{5}$$

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 外部影响

    然而，我们尚未涵盖所有情况。可能存在一些**与状态本身无关**的变化——外部世界可能正在影响系统。

    例如，如果状态模拟的是火车的运动，火车司机可能推动油门，导致火车加速。类似地，在我们的机器人示例中，导航软件可能发出指令让车轮转向或停止。如果我们掌握了关于外部世界正在发生什么的这些额外信息，我们可以将其放入一个名为 $\color{darkorange}{\vec{\mathbf{u}}_{k}}$ 的向量中，对其进行处理，并将其作为修正项添加到我们的预测中。

    假设我们已知由于油门设定或控制指令产生的预期加速度 $\color{darkorange}{a}$，形成了匀加速直线运动。那么根据基本运动学，我们得到：

    $$
    \begin{aligned}
    \color{deeppink}{p_k} &= \color{royalblue}{p_{k-1}} + & \color{royalblue}{\Delta t}{v_{k-1}} + & \color{darkorange}\frac{1}{2}{a} {\Delta t}^2 \\
    \color{deeppink}{v_k} &= &\color{royalblue}{v_{k-1}} + & \color{darkorange}{a} {\Delta t}
    \end{aligned}
    $$

    以矩阵形式表示：

    $$\begin{equation}
    \begin{split}
    \color{deeppink}{\mathbf{\hat{x}}_k} &= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}} + \color{darkorange}\begin{bmatrix}
    \frac{\Delta t^2}{2} \\
    \Delta t
    \end{bmatrix} \color{darkorange}{a} \\
    &= \mathbf{F}_k \color{royalblue}{\mathbf{\hat{x}}_{k-1}} + \color{darkorange}\mathbf{B}_k \color{darkorange}{\vec{\mathbf{u}_k}}
    \end{split}
    \end{equation}\tag{6}$$

    $\mathbf{B}_k$ 称为**控制矩阵**，$\color{darkorange}{\vec{\mathbf{u}}_{k}}$ 称为**控制向量**。（对于没有外部影响的非常简单系统，可以省略这些）。

    让我们再补充一个细节。如果我们的预测并非对实际发生情况的 100% 准确模型，会怎样？
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 外部不确定性

    如果状态根据其自身属性演化，一切都没有问题。如果状态根据外部力演化，只要我们知晓这些外部力是什么，一切*仍然*没有问题。

    但是，对于那些我们*不知道*的力呢？

    例如，如果我们正在跟踪一架四旋翼飞行器，它可能受到风的冲击。如果我们正在跟踪一个轮式机器人，车轮可能打滑，或者地面上的凸起可能使其减速。我们无法跟踪这些事情，如果其中任何一件发生，我们的预测可能会出现偏差，因为我们没有考虑那些额外的力。

    我们可以通过在每个预测步骤之后添加一些新的不确定性，来建模与"世界"（即我们没有跟踪的事物）相关的不确定性。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    我们原始估计中的每个状态都可能移动到一个*范围*内的状态。因为我们非常喜欢高斯分布，所以我们假设 $\color{royalblue}{\mathbf{\hat{x}}_{k-1}}$ 中的每个点都被移动到某个具有协方差 $\color{mediumaquamarine}{\mathbf{Q}_k}$ 的高斯分布内部。换言之，我们将未跟踪的影响视为具有协方差 $\color{mediumaquamarine}{\mathbf{Q}_k}$ 的**噪声**。

    这产生了一个新的高斯分布，具有不同的协方差（但均值相同）：

    我们通过简单地**加上** ${\color{mediumaquamarine}{\mathbf{Q}_k}}$ 来获得扩展后的协方差，从而给出**预测步骤**的完整表达式：

    $$\begin{equation}
    \begin{split}
    \color{deeppink}{\mathbf{\hat{x}}_k} &= \mathbf{F}_k {\color{royalblue}{\mathbf{\hat{x}}_{k-1}}}  +  \mathbf{B}_k \color{darkorange}{\vec{\mathbf{u}_k}} \\
    \color{deeppink}{\mathbf{P}_k} &= \mathbf{F_k} {\color{royalblue}{\mathbf{P}_{k-1}}} \mathbf{F}_k^T + \color{mediumaquamarine}{\mathbf{Q}_k}
    \end{split}
    \end{equation}\tag{7}$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    换言之，**新的最佳估计**$\mathbf{\hat{x}}_{k}$是根据**先前的最佳估计**${\mathbf{\hat{x}}_{k-1}}$做出的**预测**，加上对**已知外部影响**的**修正**。

    而**新的不确定性**是根据**旧的不确定性**进行**预测**的，并附加了**来自环境的额外不确定性**。

    好的，这足够简单了。我们得到了一个关于系统可能位置的模糊估计，由 $\color{deeppink}{\mathbf{\hat{x}}_k}$ 和 $\color{deeppink}{\mathbf{P}_k}$ 给出。

    当我们从传感器获得一些数据时，会发生什么呢？

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 用测量值修正估计

    我们可能拥有多个传感器，它们提供关于系统状态的信息。

    目前它们测量什么并不重要；也许一个读取位置，另一个读取速度。

    每个传感器告诉我们一些关于状态的**间接**信息——换言之，传感器对状态进行操作并产生一组**读数**。

    请注意，读数的单位和比例可能与我们跟踪的状态的单位和比例不同。

    因此，我们需要将状态估计得到的数值通过一个变换来将它们映射到传感器坐标系下的数值。我们将用一个矩阵 $\mathbf{H_k}$ 来建模这种状态估计坐标系到传感器坐标系的变换。
    """)
    return


@app.cell(hide_code=True)
def _(ConnectionPatch, FancyArrowPatch, figure_as_svg, np, pe, plt):
    def _():
        # =========================
        # 二维高斯密度函数
        # =========================
        def gaussian_pdf_2d(grid, mu, cov):
            diff = grid - mu
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)

            exponent = -0.5 * np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
            norm = 1.0 / (2 * np.pi * np.sqrt(det_cov))

            return norm * np.exp(exponent)


        # =========================
        # 左图坐标系：与参考图 2 完全对齐
        # 横轴：位置 p
        # 纵轴：速度 v
        # =========================
        p_min, p_max = -8, 8
        v_min, v_max = -15, 1

        p = np.linspace(p_min, p_max, 700)
        v = np.linspace(v_min, v_max, 700)
        P_grid, V_grid = np.meshgrid(p, v)

        state_grid = np.stack([P_grid, V_grid], axis=-1)


        # =========================
        # 写死的示意数值
        # 状态：x = [position, velocity]^T
        # 传感器读数：z = H_k x
        # =========================
        x_hat = np.array([-0.2, -7.0])

        P_k = np.array([
            [2.10, -1.85],
            [-1.85, 3.10],
        ])

        H_k = np.array([
            [0.58, 0.10],
            [-0.18, 0.42],
        ])

        z_hat = H_k @ x_hat
        R_k = H_k @ P_k @ H_k.T


        # =========================
        # 左图：状态空间中的分布
        # =========================
        Z_state = gaussian_pdf_2d(state_grid, x_hat, P_k)
        Z_state = Z_state / Z_state.max()


        # =========================
        # 右图：传感器坐标系中的分布
        # =========================
        s1_min, s1_max = -4.5, 4.5
        s2_min, s2_max = -4.5, 4.5

        s1 = np.linspace(s1_min, s1_max, 600)
        s2 = np.linspace(s2_min, s2_max, 600)
        S1_grid, S2_grid = np.meshgrid(s1, s2)

        sensor_grid = np.stack([S1_grid, S2_grid], axis=-1)

        Z_sensor = gaussian_pdf_2d(sensor_grid, z_hat, R_k)
        Z_sensor = Z_sensor / Z_sensor.max()


        # =========================
        # 绘图布局
        # =========================
        fig, (ax_state, ax_sensor) = plt.subplots(
            1,
            2,
            figsize=(14, 7.5),
            dpi=150,
            gridspec_kw={"width_ratios": [1.05, 0.95], "wspace": 0.22},
        )

        fig.patch.set_facecolor("white")

        cmap = "viridis"


        # =========================
        # 左图：状态估计坐标系
        # =========================
        im_state = ax_state.imshow(
            Z_state,
            extent=[p_min, p_max, v_min, v_max],
            origin="lower",
            cmap=cmap,
            aspect="auto",
            vmin=0.0,
            vmax=1.05,
        )

        ax_state.contour(
            P_grid,
            V_grid,
            Z_state,
            levels=[0.12, 0.25, 0.40, 0.58, 0.76],
            colors="white",
            linewidths=1.45,
            alpha=0.55,
        )

        ax_state.axvline(0, linestyle="--", linewidth=1.5, color="white", alpha=0.82)
        ax_state.axhline(-7, linestyle="--", linewidth=1.5, color="white", alpha=0.82)

        ax_state.set_xlim(p_min, p_max)
        ax_state.set_ylim(v_min, v_max)

        ax_state.set_xticks(np.arange(-8, 8, 2))
        ax_state.set_yticks(np.arange(0, -16, -2))

        ax_state.set_xlabel("位置 p", fontsize=15)
        ax_state.set_ylabel("速度 v", fontsize=15)
        ax_state.tick_params(labelsize=12)


        # =========================
        # 右图：传感器读数坐标系
        # =========================
        im_sensor = ax_sensor.imshow(
            Z_sensor,
            extent=[s1_min, s1_max, s2_min, s2_max],
            origin="lower",
            cmap=cmap,
            aspect="auto",
            vmin=0.0,
            vmax=1.05,
        )

        ax_sensor.contour(
            S1_grid,
            S2_grid,
            Z_sensor,
            levels=[0.12, 0.25, 0.40, 0.58, 0.76],
            colors="white",
            linewidths=1.45,
            alpha=0.60,
        )

        ax_sensor.axvline(0, linestyle="--", linewidth=1.5, color="white", alpha=0.82)
        ax_sensor.axhline(0, linestyle="--", linewidth=1.5, color="white", alpha=0.82)

        ax_sensor.set_xlim(s1_min, s1_max)
        ax_sensor.set_ylim(s2_min, s2_max)

        ax_sensor.set_xticks(np.arange(-4, 5, 2))
        ax_sensor.set_yticks(np.arange(-4, 5, 2))

        ax_sensor.set_xlabel("传感器 1 读数", fontsize=15)
        ax_sensor.set_ylabel("传感器 2 读数", fontsize=15)
        ax_sensor.tick_params(labelsize=12)


        # =========================
        # 文字描边
        # =========================
        white_stroke = [
            pe.withStroke(linewidth=5, foreground="black", alpha=0.85)
        ]

        black_stroke = [
            pe.withStroke(linewidth=4, foreground="white", alpha=0.90)
        ]


        # =========================
        # 均值点与分布标注
        # =========================
        ax_state.scatter(
            x_hat[0],
            x_hat[1],
            s=62,
            color="white",
            edgecolor="black",
            linewidth=1.1,
            zorder=10,
        )

        ax_sensor.scatter(
            z_hat[0],
            z_hat[1],
            s=62,
            color="white",
            edgecolor="black",
            linewidth=1.1,
            zorder=10,
        )

        ax_state.text(
            -6.9,
            -12.6,
            "状态估计坐标系\nx_hat k, P k",
            color="white",
            fontsize=17,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.50, edgecolor="none"),
            zorder=12,
        )

        ax_sensor.text(
            -3.95,
            3.25,
            "传感器读数坐标系\nz_hat k, S k",
            color="white",
            fontsize=17,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.50, edgecolor="none"),
            zorder=12,
        )


        # =========================
        # 在两个坐标系中画几个对应点
        # 表示 H_k 对每一个可能状态点进行映射
        # =========================
        sample_points_state = np.array([
            x_hat + [-1.20,  1.15],
            x_hat + [ 0.95, -1.10],
            x_hat + [-0.55, -0.85],
            x_hat + [ 1.25,  0.55],
        ])

        sample_points_sensor = (H_k @ sample_points_state.T).T

        for xs, zs in zip(sample_points_state, sample_points_sensor):
            ax_state.scatter(
                xs[0],
                xs[1],
                s=28,
                color="white",
                edgecolor="white",
                linewidth=0.8,
                alpha=0.75,
                zorder=9,
            )

            ax_sensor.scatter(
                zs[0],
                zs[1],
                s=28,
                color="white",
                edgecolor="white",
                linewidth=0.8,
                alpha=0.80,
                zorder=9,
            )

            ax_state.plot(
                [x_hat[0], xs[0]],
                [x_hat[1], xs[1]],
                color="white",
                linewidth=1.6,
                alpha=0.75,
                zorder=8,
            )

            ax_sensor.plot(
                [z_hat[0], zs[0]],
                [z_hat[1], zs[1]],
                color="white",
                linewidth=1.6,
                alpha=0.75,
                zorder=8,
            )


        # =========================
        # 跨坐标系的映射连线
        # =========================
        for xs, zs in zip(sample_points_state, sample_points_sensor):
            con = ConnectionPatch(
                xyA=xs,
                xyB=zs,
                coordsA="data",
                coordsB="data",
                axesA=ax_state,
                axesB=ax_sensor,
                arrowstyle="-",
                linewidth=1.2,
                color="black",
                alpha=0.22,
                zorder=2,
            )
            fig.add_artist(con)


        # =========================
        # 中间大箭头与 H_k 标注
        # =========================
        arrow = FancyArrowPatch(
            (0.455, 0.21),
            (0.575, 0.21),
            transform=fig.transFigure,
            arrowstyle="->",
            mutation_scale=26,
            linewidth=3.0,
            color="green",
            alpha=0.82,
            connectionstyle="arc3,rad=0.08",
            zorder=20,
        )

        fig.add_artist(arrow)

        fig.text(
            0.515,
            0.255,
            r"$\mathbf{H}_k$",
            fontsize=30,
            weight="bold",
            ha="center",
            va="center",
            color="black",
            path_effects=black_stroke,
            zorder=21,
        )


        # =========================
        # 公式标注
        # =========================
        ax_sensor.text(
            1.05,
            -3.55,
            r"$\hat{\mathbf{z}}_k=\mathbf{H}_k\hat{\mathbf{x}}_k$" "\n"
            r"$\mathbf{S}_k=\mathbf{H}_k\mathbf{P}_k\mathbf{H}_k^\mathsf{T}$",
            color="white",
            fontsize=16,
            weight="bold",
            path_effects=white_stroke,
            linespacing=1.35,
            zorder=12,
        )


        # =========================
        # 共享 colorbar
        # =========================
        cbar = fig.colorbar(
            im_sensor,
            ax=[ax_state, ax_sensor],
            fraction=0.035,
            pad=0.035,
        )

        cbar.set_label("概率密度（示意）", rotation=90, labelpad=18, fontsize=14)
        cbar.ax.tick_params(labelsize=12)
        return fig


    figure_as_svg(_())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    根据我们的预测结果，我们期望看到的传感器读数分布如下所示：

    $$\begin{equation}
    \begin{aligned}
    \vec{\mu}_{\text{expected}} &= \mathbf{H}_k \color{deeppink}{\mathbf{\hat{x}}_k} \\
    \mathbf{\Sigma}_{\text{expected}} &= \mathbf{H}_k \color{deeppink}{\mathbf{P}_k} \mathbf{H}_k^T
    \end{aligned}
    \end{equation}\tag{8}$$

    从每个观察到的读数中，我们可能推测系统处于某一特定状态。但由于存在不确定性，**某些状态比其他状态**更有可能产生我们看到的读数。

    我们将这种不确定性（即传感器噪声）的**协方差**称为 $\color{mediumaquamarine}{\mathbf{R}_k}$。该分布的**均值**等于我们观察到的读数，我们将其记为 $\color{yellowgreen}{\vec{\mathbf{z}}_{k}}$。

    类似地，我们将传感器的这种不确定性也建模为高斯分布。
    """)
    return


@app.cell(hide_code=True)
def _(figure_as_svg, np, pe, plt):
    def _():
        # =========================
        # 二维高斯密度函数
        # =========================
        def gaussian_pdf_2d(grid, mu, cov):
            diff = grid - mu
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)

            exponent = -0.5 * np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
            norm = 1.0 / (2 * np.pi * np.sqrt(det_cov))

            return norm * np.exp(exponent)


        # =========================
        # 传感器读数坐标系（示意）
        # 这里的数值全部写死，只用于配图
        # =========================
        s1_min, s1_max = -5, 5
        s2_min, s2_max = -5, 5

        s1 = np.linspace(s1_min, s1_max, 700)
        s2 = np.linspace(s2_min, s2_max, 700)
        S1_grid, S2_grid = np.meshgrid(s1, s2)
        sensor_grid = np.stack([S1_grid, S2_grid], axis=-1)

        # 观测到的读数 z_k（均值）
        z_k = np.array([0.8, -1.1])

        # 传感器噪声协方差 R_k
        R_k = np.array([
            [1.35, -0.65],
            [-0.65, 0.95],
        ])

        Z = gaussian_pdf_2d(sensor_grid, z_k, R_k)
        Z = Z / Z.max()  # 归一化，仅用于示意图显示


        # =========================
        # 绘图
        # =========================
        fig, ax = plt.subplots(figsize=(10.5, 8), dpi=150)

        im = ax.imshow(
            Z,
            extent=[s1_min, s1_max, s2_min, s2_max],
            origin="lower",
            cmap="viridis",
            aspect="auto",
            vmin=0.0,
            vmax=1.05,
        )

        # 白色等高线，表现“高斯分布”的形状
        ax.contour(
            S1_grid,
            S2_grid,
            Z,
            levels=[0.12, 0.25, 0.40, 0.58, 0.76],
            colors="white",
            linewidths=1.6,
            alpha=0.62,
        )

        # 坐标轴参考虚线
        ax.axvline(z_k[0], linestyle="--", linewidth=1.5, color="white", alpha=0.82)
        ax.axhline(z_k[1], linestyle="--", linewidth=1.5, color="white", alpha=0.82)

        # 均值点（观测到的读数）
        ax.scatter(
            z_k[0],
            z_k[1],
            s=64,
            color="white",
            edgecolor="black",
            linewidth=1.1,
            zorder=10,
        )

        # 描边效果，便于白字压在彩色背景上
        stroke = [pe.withStroke(linewidth=5, foreground="black", alpha=0.85)]

        # 标注 z_k
        ax.annotate(
            "观测读数（均值）\nz_k",
            xy=z_k,
            xytext=(1.9, 1.35),
            color="white",
            fontsize=17,
            weight="bold",
            arrowprops=dict(
                arrowstyle="->",
                color="white",
                lw=2.3,
                shrinkA=3,
                shrinkB=3,
            ),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.50, edgecolor="none"),
            zorder=12,
        )

        # 标注 R_k
        ax.text(
            -4.15,
            -3.95,
            "传感器噪声协方差\nR_k",
            color="white",
            fontsize=18,
            weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.50, edgecolor="none"),
            zorder=12,
        )

        # 用箭头指向椭圆区域，表示协方差描述不确定性形状
        ax.annotate(
            "",
            xy=(-0.15, -1.75),
            xytext=(-2.45, -3.0),
            arrowprops=dict(
                arrowstyle="->",
                color="white",
                lw=2.4,
                connectionstyle="arc3,rad=-0.25",
            ),
            zorder=12,
        )

        # 右下角补一行说明
        ax.text(
            1.15,
            -4.25,
            r"$\mathbf{z}_k \sim \mathcal{N}(\vec{\mathbf{z}}_k,\ \mathbf{R}_k)$",
            color="white",
            fontsize=16,
            weight="bold",
            path_effects=stroke,
            zorder=12,
        )

        # 坐标轴设置
        ax.set_xlim(s1_min, s1_max)
        ax.set_ylim(s2_min, s2_max)

        ax.set_xticks(np.arange(-4, 5, 2))
        ax.set_yticks(np.arange(-4, 5, 2))

        ax.set_xlabel("传感器 1 读数", fontsize=15)
        ax.set_ylabel("传感器 2 读数", fontsize=15)
        ax.tick_params(labelsize=12)

        # colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("概率密度（示意）", rotation=90, labelpad=18, fontsize=14)
        cbar.ax.tick_params(labelsize=12)

        fig.tight_layout()
        return fig


    figure_as_svg(_())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    因此，我们现在有了两个高斯分布：一个围绕我们变换后的预测的均值，另一个围绕我们实际获得的传感器读数。
    """)
    return


@app.cell(hide_code=True)
def _(FancyArrowPatch, figure_as_svg, np, pe, plt):
    def gaussian_pdf_2d_for_demo(grid, mean, cov):
        diff = grid - mean
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)

        exponent = -0.5 * np.einsum("...i,ij,...j->...", diff, inv_cov, diff)
        norm = 1.0 / (2 * np.pi * np.sqrt(det_cov))

        return norm * np.exp(exponent)


    def add_handdrawn_axes_for_demo(ax, xlim, ylim, xlabel, ylabel):
        ax.set_facecolor("black")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(False)

        axis_style = dict(
            arrowstyle="->",
            mutation_scale=24,
            linewidth=3.0,
            color="white",
            shrinkA=0,
            shrinkB=0,
        )

        x0 = xlim[0] + 0.7
        y0 = ylim[0] + 0.7

        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (xlim[1] - 0.5, y0),
                **axis_style,
                zorder=20,
            )
        )

        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x0, ylim[1] - 0.5),
                **axis_style,
                zorder=20,
            )
        )

        stroke = [pe.withStroke(linewidth=4, foreground="black", alpha=0.85)]

        ax.text(
            (xlim[0] + xlim[1]) / 2,
            ylim[0] + 0.08,
            xlabel,
            color="white",
            fontsize=22,
            ha="center",
            va="bottom",
            fontstyle="italic",
            path_effects=stroke,
        )

        ax.text(
            xlim[0] + 0.05,
            (ylim[0] + ylim[1]) / 2,
            ylabel,
            color="white",
            fontsize=22,
            ha="left",
            va="center",
            rotation=90,
            fontstyle="italic",
            path_effects=stroke,
        )


    # =========================
    # 传感器读数坐标系
    # =========================
    x1_min, x1_max = -6, 6
    x2_min, x2_max = -6, 6

    x1_img1 = np.linspace(x1_min, x1_max, 700)
    x2_img1 = np.linspace(x2_min, x2_max, 700)
    X1_img1, X2_img1 = np.meshgrid(x1_img1, x2_img1)
    grid_img1 = np.stack([X1_img1, X2_img1], axis=-1)


    # 粉色：由预测状态推出的读数分布 H_k x_hat_k
    mean_pred_reading_img1 = np.array([0.55, 1.15])

    cov_pred_reading_img1 = np.array([
        [4.00, 2.10],
        [2.10, 1.70],
    ])

    # 绿色：实际传感器读数 z_k 附近的噪声分布
    mean_sensor_reading_img1 = np.array([0.75, -0.10])

    cov_sensor_reading_img1 = np.array([
        [3.10, -1.65],
        [-1.65, 1.75],
    ])

    Z_pink_img1 = gaussian_pdf_2d_for_demo(
        grid_img1,
        mean_pred_reading_img1,
        cov_pred_reading_img1,
    )
    Z_green_img1 = gaussian_pdf_2d_for_demo(
        grid_img1,
        mean_sensor_reading_img1,
        cov_sensor_reading_img1,
    )

    Z_pink_img1 = Z_pink_img1 / Z_pink_img1.max()
    Z_green_img1 = Z_green_img1 / Z_green_img1.max()


    # =========================
    # 用 RGB 叠加做黑底发光效果
    # =========================
    pink_rgb_img1 = np.array([1.00, 0.00, 0.70])
    green_rgb_img1 = np.array([0.55, 1.00, 0.00])

    pink_strength_img1 = Z_pink_img1**0.72
    green_strength_img1 = Z_green_img1**0.72

    rgb_img1 = (
        pink_strength_img1[..., None] * pink_rgb_img1 * 0.95
        + green_strength_img1[..., None] * green_rgb_img1 * 0.95
    )

    # 两个分布重叠处偏白，表示二者共同支持的区域
    overlap_img1 = (Z_pink_img1 * Z_green_img1)
    overlap_img1 = overlap_img1 / overlap_img1.max()
    rgb_img1 = rgb_img1 + overlap_img1[..., None] * np.array([0.80, 0.80, 0.80])

    rgb_img1 = np.clip(rgb_img1, 0, 1)


    # =========================
    # 绘图
    # =========================
    fig_img1, ax_img1 = plt.subplots(figsize=(8.2, 8.0), dpi=150)
    fig_img1.patch.set_facecolor("black")

    ax_img1.imshow(
        rgb_img1,
        extent=[x1_min, x1_max, x2_min, x2_max],
        origin="lower",
        aspect="auto",
    )

    add_handdrawn_axes_for_demo(
        ax_img1,
        (x1_min, x1_max),
        (x2_min, x2_max),
        "sensor 1 reading",
        "sensor 2 reading",
    )

    stroke_img1 = [pe.withStroke(linewidth=5, foreground="black", alpha=0.85)]

    # 均值点
    ax_img1.scatter(
        mean_pred_reading_img1[0],
        mean_pred_reading_img1[1],
        s=32,
        color="#ff00aa",
        edgecolor="#ff00aa",
        zorder=30,
    )

    ax_img1.scatter(
        mean_sensor_reading_img1[0],
        mean_sensor_reading_img1[1],
        s=32,
        color="#a6ff00",
        edgecolor="#a6ff00",
        zorder=30,
    )

    # 标注 H_k x_hat_k
    ax_img1.annotate(
        r"$\mathbf{H}_k\hat{\mathbf{x}}_k$",
        xy=mean_pred_reading_img1,
        xytext=(-1.05, 3.15),
        color="white",
        fontsize=27,
        weight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color="#ff00aa",
            lw=2.8,
            connectionstyle="arc3,rad=-0.08",
        ),
        path_effects=stroke_img1,
        zorder=35,
    )

    # 标注 z_k
    ax_img1.annotate(
        r"$\vec{\mathbf{z}}_k$",
        xy=mean_sensor_reading_img1,
        xytext=(2.15, -0.25),
        color="#a6ff00",
        fontsize=27,
        weight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color="#a6ff00",
            lw=2.8,
            connectionstyle="arc3,rad=-0.08",
        ),
        path_effects=stroke_img1,
        zorder=35,
    )

    # 图例：estimate 1 / estimate 2
    legend_x_img1 = -4.85
    legend_y_img1 = 5.35

    ax_img1.plot(
        [legend_x_img1, legend_x_img1 + 0.70],
        [legend_y_img1, legend_y_img1],
        color="#ff00aa",
        linewidth=4,
        solid_capstyle="round",
        zorder=35,
    )

    ax_img1.text(
        legend_x_img1 + 0.9,
        legend_y_img1 - 0.13,
        "estimate 1",
        color="white",
        fontsize=19,
        fontstyle="italic",
        weight="bold",
        path_effects=stroke_img1,
        zorder=35,
    )

    ax_img1.plot(
        [legend_x_img1, legend_x_img1 + 0.70],
        [legend_y_img1 - 0.75, legend_y_img1 - 0.75],
        color="#a6ff00",
        linewidth=4,
        solid_capstyle="round",
        zorder=35,
    )

    ax_img1.text(
        legend_x_img1 + 0.9,
        legend_y_img1 - 0.88,
        "estimate 2",
        color="white",
        fontsize=19,
        fontstyle="italic",
        weight="bold",
        path_effects=stroke_img1,
        zorder=35,
    )

    # 中间说明
    ax_img1.text(
        -4.8,
        -4.45,
        "协调两种读数推测：\n预测读数分布  ×  传感器读数分布",
        color="white",
        fontsize=15,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.50, edgecolor="none"),
        zorder=35,
    )

    fig_img1.tight_layout()
    figure_as_svg(fig_img1)
    return add_handdrawn_axes_for_demo, gaussian_pdf_2d_for_demo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    我们必须尝试协调基于预测状态对读数的推测（粉色）与基于实际传感器读数的另一种推测（绿色）。

    那么，我们新的最可能状态是什么？

    对于任何可能的读数 $(z_1, z_2)$，我们有两个关联概率：
    1. 我们的传感器读数 $\color{yellowgreen}{\vec{\mathbf{z}_k}}$ 是对 $(z_1, z_2)$ 的（误）测量的概率；
    2. 我们先前估计的概率认为 $(z_1, z_2)$ 是我们应该看到的读数。

    如果我们有两个概率，并且想知道*两者同时成立*的可能性，我们只需将它们相乘。因此，我们将两个高斯分布相乘，并且结果仍然服从高斯分布（详细见附录3）：
    """)
    return


@app.cell(hide_code=True)
def _(
    add_handdrawn_axes_for_demo,
    figure_as_svg,
    gaussian_pdf_2d_for_demo,
    np,
    pe,
    plt,
):
    # =========================
    # 两个高斯分布相乘后的共同最可能区域
    # 注意：本 cell 依赖上一个 cell 中的 import 和 gaussian_pdf_2d_for_demo()
    # =========================

    x1_min_img2, x1_max_img2 = -6, 6
    x2_min_img2, x2_max_img2 = -6, 6

    x1_img2 = np.linspace(x1_min_img2, x1_max_img2, 800)
    x2_img2 = np.linspace(x2_min_img2, x2_max_img2, 800)
    X1_img2, X2_img2 = np.meshgrid(x1_img2, x2_img2)
    grid_img2 = np.stack([X1_img2, X2_img2], axis=-1)


    # 粉色：先前估计认为“应该看到”的读数
    mean_estimate_1_img2 = np.array([0.45, 1.15])

    cov_estimate_1_img2 = np.array([
        [4.10, 2.35],
        [2.35, 1.75],
    ])

    # 绿色：实际传感器读数 z_k 可能对应的真实读数
    mean_estimate_2_img2 = np.array([0.60, -0.35])

    cov_estimate_2_img2 = np.array([
        [3.55, -2.25],
        [-2.25, 2.25],
    ])

    Z_estimate_1_img2 = gaussian_pdf_2d_for_demo(
        grid_img2,
        mean_estimate_1_img2,
        cov_estimate_1_img2,
    )

    Z_estimate_2_img2 = gaussian_pdf_2d_for_demo(
        grid_img2,
        mean_estimate_2_img2,
        cov_estimate_2_img2,
    )

    Z_estimate_1_img2 = Z_estimate_1_img2 / Z_estimate_1_img2.max()
    Z_estimate_2_img2 = Z_estimate_2_img2 / Z_estimate_2_img2.max()

    # 两个概率同时成立：相乘
    Z_product_img2 = Z_estimate_1_img2 * Z_estimate_2_img2
    Z_product_img2 = Z_product_img2 / Z_product_img2.max()


    # =========================
    # 灰色发光区域：乘积后的新分布
    # =========================
    gray_strength_img2 = Z_product_img2**0.70
    rgb_img2 = np.zeros((*gray_strength_img2.shape, 3))
    rgb_img2[..., 0] = gray_strength_img2 * 0.78
    rgb_img2[..., 1] = gray_strength_img2 * 0.78
    rgb_img2[..., 2] = gray_strength_img2 * 0.78
    rgb_img2 = np.clip(rgb_img2, 0, 1)


    # =========================
    # 绘图
    # =========================
    fig_img2, ax_img2 = plt.subplots(figsize=(8.2, 8.0), dpi=150)
    fig_img2.patch.set_facecolor("black")

    ax_img2.imshow(
        rgb_img2,
        extent=[x1_min_img2, x1_max_img2, x2_min_img2, x2_max_img2],
        origin="lower",
        aspect="auto",
    )

    add_handdrawn_axes_for_demo(
        ax_img2,
        (x1_min_img2, x1_max_img2),
        (x2_min_img2, x2_max_img2),
        "sensor 1 reading",
        "sensor 2 reading",
    )

    stroke_img2 = [pe.withStroke(linewidth=5, foreground="black", alpha=0.85)]

    # 粉色虚线椭圆：estimate 1
    ax_img2.contour(
        X1_img2,
        X2_img2,
        Z_estimate_1_img2,
        levels=[0.18],
        colors=["#ff00aa"],
        linewidths=2.2,
        linestyles="--",
        alpha=0.95,
        zorder=20,
    )

    # 绿色虚线椭圆：estimate 2
    ax_img2.contour(
        X1_img2,
        X2_img2,
        Z_estimate_2_img2,
        levels=[0.18],
        colors=["#a6ff00"],
        linewidths=2.2,
        linestyles="--",
        alpha=0.95,
        zorder=20,
    )

    # 乘积后的等高线
    ax_img2.contour(
        X1_img2,
        X2_img2,
        Z_product_img2,
        levels=[0.25, 0.50, 0.75],
        colors="white",
        linewidths=1.2,
        alpha=0.45,
        zorder=22,
    )

    # 新的最可能读数位置
    max_index_img2 = np.unravel_index(np.argmax(Z_product_img2), Z_product_img2.shape)
    new_best_reading_img2 = np.array([
        X1_img2[max_index_img2],
        X2_img2[max_index_img2],
    ])

    ax_img2.scatter(
        new_best_reading_img2[0],
        new_best_reading_img2[1],
        s=42,
        color="white",
        edgecolor="black",
        linewidth=1.0,
        zorder=30,
    )

    ax_img2.annotate(
        "estimate 1\nAND\nestimate 2",
        xy=new_best_reading_img2,
        xytext=(-3.9, 3.45),
        color="white",
        fontsize=19,
        ha="center",
        va="center",
        fontstyle="italic",
        weight="bold",
        arrowprops=dict(
            arrowstyle="->",
            color="white",
            lw=2.7,
            connectionstyle="arc3,rad=0.35",
        ),
        path_effects=stroke_img2,
        zorder=35,
    )

    ax_img2.text(
        -4.65,
        -4.65,
        r"$p_{\mathrm{new}}(z_1,z_2)$" "\n"
        r"$\propto$ estimate 1 $\times$ estimate 2",
        color="white",
        fontsize=17,
        weight="bold",
        path_effects=stroke_img2,
        zorder=35,
    )

    ax_img2.text(
        1.15,
        4.45,
        "两个高斯分布相乘后，\n重叠区域变成新的高概率区域",
        color="white",
        fontsize=15,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", alpha=0.50, edgecolor="none"),
        zorder=35,
    )

    fig_img2.tight_layout()
    figure_as_svg(fig_img2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    我们得到的是**重叠区域**，即*两个*分布都明亮/高亮的区域。它比我们先前的任何一个估计都精确得多。该分布的均值是**两个**估计都最可能成立的配置，因此是给定我们拥有的所有信息时对真实配置的**最佳推测**。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 高斯分布的融合

    下面先用一维高斯分布证明两个高斯分布的成绩仍然是高斯分布。两个一维高斯分布看起来像这样：

    $$\mathcal{N}(x, \mu, \sigma) = \frac{1}{ \sigma \sqrt{2 \pi} } e^{ -\frac{1}{2}\frac{(x - \mu)^2}{\sigma^2} }\tag{9}$$

    当你将两个高斯函数相乘时，会发生什么？
    $$\mathcal{N}(x, {\color{fuchsia}{\mu_0}}, {\color{deeppink}{\sigma_0}}) \cdot \mathcal{N}(x, \color{yellowgreen}{\mu_1}, \color{mediumaquamarine}{\sigma_1}) \stackrel{?}{=} \mathcal{N}(x, \color{royalblue}{\mu'}, \color{mediumblue}{\sigma'} )\tag{10}$$

    你可以将公式9代入公式10，得到：

    $$\begin{equation}
    \begin{aligned}
    \color{royalblue}{\mu'} &= \mu_0 + \frac{\sigma_0^2 (\mu_1 - \mu_0)}{\sigma_0^2 + \sigma_1^2} \\
    \color{mediumblue}{\sigma'}^2 &= \sigma_0^2 - \frac{\sigma_0^4}{\sigma_0^2 + \sigma_1^2}
    \end{aligned}
    \end{equation}\tag{11}$$

    我们可以通过提取一个公共项并将其命名为 $\color{purple}{\mathbf{k}}$ 来简化：

    $$
    \color{purple}{\mathbf{k}} = \frac{\sigma_0^2}{\sigma_0^2 + \sigma_1^2} \tag{12}$$

    $$\begin{equation}
    \begin{split}
    \color{royalblue}{\mu'} &= \mu_0 + \color{purple}{\mathbf{k}} (\mu_1 – \mu_0)\\
    \color{mediumblue}{\sigma'}^2 &= \sigma_0^2 – \color{purple}{\mathbf{k}} \sigma_0^2
    \end{split} \tag{13}
    \end{equation}$$

    请注意，根据公式13所示，你可以如何取先前的估计（$\mu_{0},\sigma^{2}_{0}$）并**加上某项**来得到新的估计$\mu^{\prime},\sigma^{\prime 2}$。并且看看这个公式多么简洁！
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    那么矩阵版本呢？

    根据附录3中的推导可知：

    $$
    \boxed{
    \mu
    =
    (\Sigma_1^{-1}+\Sigma_2^{-1})^{-1}
    (\Sigma_1^{-1}\mu_1+\Sigma_2^{-1}\mu_2)
    }
    $$

    $$
    \boxed{
    \Sigma
    =
    (\Sigma_1^{-1}+\Sigma_2^{-1})^{-1}
    }
    $$

    好吧，让我们将方程12和13改写为矩阵形式。如果 $\Sigma$ 是高斯分布的协方差矩阵，$\vec{\mu}$ 是其沿各轴的均值，那么：

    $$\begin{equation} \tag{14}
    \color{purple}{\mathbf{K}} = \Sigma_0 (\Sigma_0 + \Sigma_1)^{-1}
    \end{equation}$$

    $$\begin{equation}
    \begin{aligned}
    \color{royalblue}{\vec{\mu}'} &= \vec{\mu_0} + &\color{purple}{\mathbf{K}} (\vec{\mu_1} – \vec{\mu_0})\\
    \color{mediumblue}{\Sigma'} &= \Sigma_0 – &\color{purple}{\mathbf{K}} \Sigma_0
    \end{aligned} \tag{15}
    \end{equation}$$

    $\color{purple}{\mathbf{K}}$ 是一个称为**卡尔曼增益**的矩阵，我们马上就会用到它。
    """)
    return


@app.cell
def _(mo):
    robot_process_std = mo.ui.slider(
        0.02,
        0.80,
        step=0.02,
        value=0.18,
        label="真实运动扰动 / 过程噪声 σₐ",
        show_value=True,
        full_width=True,
    )
    robot_measurement_std = mo.ui.slider(
        0.5,
        10.0,
        step=0.5,
        value=4.0,
        label="GPS 测量噪声 σᵣ",
        show_value=True,
        full_width=True,
    )
    robot_seed = mo.ui.number(
        start=0,
        stop=999,
        step=1,
        value=7,
        label="随机种子",
    )
    mo.vstack(
        [
            robot_process_std,
            robot_measurement_std,
            robot_seed,
        ],
        gap=0.8,
    )
    return robot_measurement_std, robot_process_std, robot_seed


@app.cell
def _(
    figure_as_svg,
    plot_robot_demo,
    robot_measurement_std,
    robot_process_std,
    robot_seed,
):
    robot_figure = plot_robot_demo(
        robot_process_std.value,
        robot_measurement_std.value,
        robot_seed.value,
    )
    figure_as_svg(robot_figure)
    return


@app.cell
def _(mo):
    mo.md(r"""
    当 GPS 很嘈杂时，滤波器更多地信任连续运动模型；当 GPS 很精确时，位置方向的卡尔曼增益
    会变大，估计会更快靠近测量。滤波器只需要保存上一时刻的状态和协方差，因此内存占用低，
    很适合实时系统与嵌入式设备。

    ---

    ## 2. 状态不是一个点，而是一个概率分布

    实际的位置和速度未知。我们只能说某些 $(p,v)$ 组合比其他组合更可能。
    线性卡尔曼滤波器用一个多元高斯分布表示这种认识：

    \[
    \mathbf{x}_k \sim \mathcal{N}(\hat{\mathbf{x}}_k, \mathbf{P}_k).
    \]

    其中 $\hat{\mathbf{x}}_k$ 是最佳估计（均值），$\mathbf{P}_k$ 是协方差矩阵：

    \[
    \hat{\mathbf{x}}_k =
    \begin{bmatrix}\hat p_k\\ \hat v_k\end{bmatrix},
    \qquad
    \mathbf{P}_k =
    \begin{bmatrix}
    \sigma_p^2 & \sigma_{pv}\\
    \sigma_{vp} & \sigma_v^2
    \end{bmatrix}.
    \]

    对角线元素是每个变量的方差；非对角线元素是协方差。协方差让一次位置测量也能间接修正
    速度，反之亦然。拖动相关系数，观察概率椭圆如何旋转。
    """)
    return


@app.cell
def _(mo):
    state_mean_position = mo.ui.slider(
        -2.0, 2.0, step=0.2, value=0.4, label="位置均值 μₚ", show_value=True
    )
    state_mean_velocity = mo.ui.slider(
        -2.0, 2.0, step=0.2, value=0.8, label="速度均值 μᵥ", show_value=True
    )
    state_std_position = mo.ui.slider(
        0.3, 1.8, step=0.1, value=1.0, label="位置标准差 σₚ", show_value=True
    )
    state_std_velocity = mo.ui.slider(
        0.3, 1.8, step=0.1, value=0.8, label="速度标准差 σᵥ", show_value=True
    )
    state_rho = mo.ui.slider(
        -0.9,
        0.9,
        step=0.05,
        value=0.65,
        label="相关系数 ρ",
        show_value=True,
    )
    mo.vstack(
        [
            mo.hstack(
                [state_mean_position, state_mean_velocity],
                widths="equal",
                gap=1.2,
            ),
            mo.hstack(
                [state_std_position, state_std_velocity],
                widths="equal",
                gap=1.2,
            ),
            state_rho,
        ],
        gap=0.8,
    )
    return (
        state_mean_position,
        state_mean_velocity,
        state_rho,
        state_std_position,
        state_std_velocity,
    )


@app.cell
def _(
    figure_as_svg,
    plot_state_distribution,
    state_mean_position,
    state_mean_velocity,
    state_rho,
    state_std_position,
    state_std_velocity,
):
    state_figure = plot_state_distribution(
        state_mean_position.value,
        state_mean_velocity.value,
        state_std_position.value,
        state_std_velocity.value,
        state_rho.value,
    )
    figure_as_svg(state_figure)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - 当 $\rho=0$ 时，位置和速度不相关，椭圆轴与坐标轴对齐。
    - 当 $|\rho|$ 增大时，知道一个变量会提供更多关于另一个变量的信息。
    - 协方差矩阵是对称的：$\sigma_{pv}=\sigma_{vp}$。

    这正是卡尔曼滤波器能够“从测量中榨取更多信息”的关键。

    ---

    ## 3. 用矩阵预测下一状态

    对匀速直线运动，

    \[
    \begin{aligned}
    p_k &= p_{k-1}+\Delta t\,v_{k-1},\\
    v_k &= v_{k-1}.
    \end{aligned}
    \]

    写成矩阵形式：

    \[
    \hat{\mathbf{x}}_k^- =
    \underbrace{\begin{bmatrix}1&\Delta t\\0&1\end{bmatrix}}_{\mathbf F_k}
    \hat{\mathbf{x}}_{k-1}^+
    +
    \underbrace{\begin{bmatrix}\frac12\Delta t^2\\\Delta t\end{bmatrix}}_{\mathbf B_k}
    a_k.
    \]

    上标 $-$ 表示测量更新前的**先验**，上标 $+$ 表示测量更新后的**后验**。
    如果随机向量经过线性变换 $\mathbf A$，其协方差满足

    \[
    \operatorname{Cov}(\mathbf A\mathbf x)
    =\mathbf A\operatorname{Cov}(\mathbf x)\mathbf A^T.
    \]

    因此，不考虑过程噪声时：

    \[
    \mathbf P_k^-=\mathbf F_k\mathbf P_{k-1}^+\mathbf F_k^T.
    \]
    """)
    return


@app.cell
def _(mo):
    prediction_dt = mo.ui.slider(
        0.2,
        3.0,
        step=0.1,
        value=1.0,
        label="时间间隔 Δt",
        show_value=True,
    )
    prediction_acceleration = mo.ui.slider(
        -1.5,
        1.5,
        step=0.1,
        value=0.4,
        label="已知控制加速度 a",
        show_value=True,
    )
    prediction_rho = mo.ui.slider(
        -0.8,
        0.8,
        step=0.1,
        value=0.4,
        label="初始位置—速度相关性",
        show_value=True,
    )
    mo.vstack(
        [prediction_dt, prediction_acceleration, prediction_rho],
        gap=0.8,
    )
    return prediction_acceleration, prediction_dt, prediction_rho


@app.cell
def _(
    figure_as_svg,
    plot_prediction,
    prediction_acceleration,
    prediction_dt,
    prediction_rho,
):
    prediction_figure = plot_prediction(
        prediction_dt.value,
        prediction_acceleration.value,
        prediction_rho.value,
    )
    figure_as_svg(prediction_figure)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $\mathbf F_k$ 不只移动分布中心，也会拉伸、旋转整个不确定性椭圆。
    $\mathbf B_k\mathbf u_k$ 则描述已知的外部控制，例如油门、转向或期望加速度。

    ---

    ## 4. 外部不确定性：过程噪声

    真实世界里还有未建模的风、打滑和地面冲击。我们把这些影响建模为零均值高斯噪声，
    其协方差为 $\mathbf Q_k$。完整预测步骤为

    \[
    \boxed{
    \begin{aligned}
    \hat{\mathbf{x}}_k^- &=
    \mathbf F_k\hat{\mathbf{x}}_{k-1}^+
    +\mathbf B_k\mathbf u_k,\\
    \mathbf P_k^- &=
    \mathbf F_k\mathbf P_{k-1}^+\mathbf F_k^T+\mathbf Q_k.
    \end{aligned}}
    \]

    对“随机加速度”模型，可以令

    \[
    \mathbf Q_k=\sigma_a^2
    \begin{bmatrix}
    \frac14\Delta t^4 & \frac12\Delta t^3\\
    \frac12\Delta t^3 & \Delta t^2
    \end{bmatrix}.
    \]

    $\mathbf Q_k$ 不改变预测均值，但会扩大预测协方差，表示我们对模型保持适当怀疑。
    """)
    return


@app.cell
def _(mo):
    noise_process_std = mo.ui.slider(
        0.0,
        1.5,
        step=0.05,
        value=0.45,
        label="未建模加速度标准差 σₐ",
        show_value=True,
    )
    noise_dt = mo.ui.slider(
        0.2,
        2.5,
        step=0.1,
        value=1.0,
        label="时间间隔 Δt",
        show_value=True,
    )
    mo.vstack([noise_process_std, noise_dt], gap=0.8)
    return noise_dt, noise_process_std


@app.cell
def _(figure_as_svg, noise_dt, noise_process_std, plot_process_noise):
    noise_figure = plot_process_noise(
        noise_process_std.value,
        noise_dt.value,
    )
    figure_as_svg(noise_figure)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. 用测量修正估计

    传感器通常只观察状态的一部分，或者使用不同的单位。观测矩阵 $\mathbf H_k$
    将状态空间映射到测量空间：

    \[
    \mathbf z_k=\mathbf H_k\mathbf x_k+\mathbf r_k,
    \qquad
    \mathbf r_k\sim\mathcal N(\mathbf 0,\mathbf R_k).
    \]

    因而，先验在测量空间中对应

    \[
    \begin{aligned}
    \boldsymbol\mu_{\text{expected}}&=\mathbf H_k\hat{\mathbf x}_k^-,\\
    \boldsymbol\Sigma_{\text{expected}}
    &=\mathbf H_k\mathbf P_k^-\mathbf H_k^T+\mathbf R_k.
    \end{aligned}
    \]

    选择传感器类型，观察同一个状态分布如何被投影到不同的测量空间。
    """)
    return


@app.cell
def _(mo):
    sensor_mode = mo.ui.dropdown(
        ["只测位置", "只测速度", "同时测位置和速度"],
        value="只测位置",
        label="观测矩阵 H 对应的传感器",
        full_width=True,
    )
    sensor_measurement_std = mo.ui.slider(
        0.1,
        2.0,
        step=0.1,
        value=0.5,
        label="测量噪声标准差",
        show_value=True,
        full_width=True,
    )
    mo.vstack([sensor_mode, sensor_measurement_std], gap=0.8)
    return sensor_measurement_std, sensor_mode


@app.cell
def _(
    figure_as_svg,
    plot_measurement_mapping,
    sensor_measurement_std,
    sensor_mode,
):
    measurement_figure = plot_measurement_mapping(
        sensor_mode.value,
        sensor_measurement_std.value,
    )
    figure_as_svg(measurement_figure)
    return


@app.cell
def _(mo):
    mo.md(r"""
    $\mathbf R_k$ 描述传感器噪声：数值越大，某个读数可能由更广泛的真实状态产生。
    现在我们拥有两个高斯分布：

    1. 模型预测在测量空间中的分布；
    2. 以实际读数 $\mathbf z_k$ 为中心、协方差为 $\mathbf R_k$ 的测量分布。

    要求二者同时成立，相当于将两个高斯概率密度相乘。其重叠区域仍然是高斯分布，
    并且通常比任一输入分布更集中。

    ---

    ## 6. 高斯分布的融合与卡尔曼增益

    一维高斯分布为

    \[
    \mathcal N(x;\mu,\sigma^2)=
    \frac{1}{\sigma\sqrt{2\pi}}
    \exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right].
    \]

    设预测为 $(\mu_0,\sigma_0^2)$、测量为 $(\mu_1,\sigma_1^2)$，则

    \[
    K=\frac{\sigma_0^2}{\sigma_0^2+\sigma_1^2},
    \]

    \[
    \mu^+=\mu_0+K(\mu_1-\mu_0),
    \qquad
    (\sigma^+)^2=(1-K)\sigma_0^2.
    \]

    $K$ 就是一维卡尔曼增益。测量方差越小，$K$ 越接近 1，后验越靠近测量；
    预测方差越小，$K$ 越接近 0，后验越靠近预测。
    """)
    return


@app.cell
def _(mo):
    fusion_prior_mean = mo.ui.slider(
        -4.0, 4.0, step=0.2, value=-1.0, label="预测均值 μ₀", show_value=True
    )
    fusion_prior_std = mo.ui.slider(
        0.2, 3.0, step=0.1, value=1.4, label="预测标准差 σ₀", show_value=True
    )
    fusion_measurement_mean = mo.ui.slider(
        -4.0, 4.0, step=0.2, value=1.5, label="测量均值 μ₁", show_value=True
    )
    fusion_measurement_std = mo.ui.slider(
        0.2, 3.0, step=0.1, value=0.8, label="测量标准差 σ₁", show_value=True
    )
    mo.vstack(
        [
            mo.hstack(
                [fusion_prior_mean, fusion_prior_std],
                widths="equal",
                gap=1.0,
            ),
            mo.hstack(
                [fusion_measurement_mean, fusion_measurement_std],
                widths="equal",
                gap=1.0,
            ),
        ],
        gap=0.8,
    )
    return (
        fusion_measurement_mean,
        fusion_measurement_std,
        fusion_prior_mean,
        fusion_prior_std,
    )


@app.cell
def _(
    figure_as_svg,
    fusion_measurement_mean,
    fusion_measurement_std,
    fusion_prior_mean,
    fusion_prior_std,
    plot_gaussian_fusion_1d,
):
    fusion_1d_figure = plot_gaussian_fusion_1d(
        fusion_prior_mean.value,
        fusion_prior_std.value,
        fusion_measurement_mean.value,
        fusion_measurement_std.value,
    )
    figure_as_svg(fusion_1d_figure)
    return


@app.cell
def _(mo):
    mo.md(r"""
    多维情况下，用均值向量和协方差矩阵替换标量即可。两个高斯分布直接融合时，

    \[
    \mathbf\Sigma^+
    =(\mathbf\Sigma_0^{-1}+\mathbf\Sigma_1^{-1})^{-1},
    \qquad
    \boldsymbol\mu^+
    =\mathbf\Sigma^+
    (\mathbf\Sigma_0^{-1}\boldsymbol\mu_0+
    \mathbf\Sigma_1^{-1}\boldsymbol\mu_1).
    \]

    改变两个分布的方向和中心距离，观察后验如何落在重叠区域中。
    """)
    return


@app.cell
def _(mo):
    fusion_2d_separation = mo.ui.slider(
        0.0,
        5.0,
        step=0.2,
        value=2.4,
        label="两个均值的距离",
        show_value=True,
    )
    fusion_2d_prior_rho = mo.ui.slider(
        -0.8,
        0.8,
        step=0.1,
        value=0.7,
        label="预测分布相关性",
        show_value=True,
    )
    fusion_2d_measurement_rho = mo.ui.slider(
        -0.8,
        0.8,
        step=0.1,
        value=-0.7,
        label="测量分布相关性",
        show_value=True,
    )
    mo.vstack(
        [
            fusion_2d_separation,
            fusion_2d_prior_rho,
            fusion_2d_measurement_rho,
        ],
        gap=0.8,
    )
    return fusion_2d_measurement_rho, fusion_2d_prior_rho, fusion_2d_separation


@app.cell
def _(
    figure_as_svg,
    fusion_2d_measurement_rho,
    fusion_2d_prior_rho,
    fusion_2d_separation,
    plot_gaussian_fusion_2d,
):
    fusion_2d_figure = plot_gaussian_fusion_2d(
        fusion_2d_separation.value,
        fusion_2d_prior_rho.value,
        fusion_2d_measurement_rho.value,
    )
    figure_as_svg(fusion_2d_figure)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. 整合全部：标准更新方程

    定义创新（测量残差）和创新协方差：

    \[
    \mathbf y_k=\mathbf z_k-\mathbf H_k\hat{\mathbf x}_k^-,
    \qquad
    \mathbf S_k=\mathbf H_k\mathbf P_k^-\mathbf H_k^T+\mathbf R_k.
    \]

    卡尔曼增益为

    \[
    \boxed{
    \mathbf K_k=\mathbf P_k^-\mathbf H_k^T\mathbf S_k^{-1}
    }.
    \]

    更新状态和协方差：

    \[
    \boxed{
    \begin{aligned}
    \hat{\mathbf x}_k^+
    &=\hat{\mathbf x}_k^-+\mathbf K_k\mathbf y_k,\\
    \mathbf P_k^+
    &=(\mathbf I-\mathbf K_k\mathbf H_k)\mathbf P_k^-.
    \end{aligned}}
    \]

    实际数值实现中，可使用更稳定的 Joseph 形式：

    \[
    \mathbf P_k^+
    =(\mathbf I-\mathbf K_k\mathbf H_k)\mathbf P_k^-
    (\mathbf I-\mathbf K_k\mathbf H_k)^T
    +\mathbf K_k\mathbf R_k\mathbf K_k^T.
    \]

    新后验会成为下一轮预测的输入，构成持续循环。
    """)
    return


@app.cell
def _(mo):
    flow_stage = mo.ui.radio(
        ["预测", "更新", "完整循环"],
        value="完整循环",
        label="高亮信息流阶段",
        inline=True,
    )
    flow_stage
    return (flow_stage,)


@app.cell
def _(figure_as_svg, flow_stage, plot_information_flow):
    flow_figure = plot_information_flow(flow_stage.value)
    figure_as_svg(flow_figure)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### 实现时真正需要的三组公式

    1. **预测均值**
       \[
       \hat{\mathbf x}_k^-=
       \mathbf F_k\hat{\mathbf x}_{k-1}^+
       +\mathbf B_k\mathbf u_k
       \]
    2. **预测协方差**
       \[
       \mathbf P_k^-=
       \mathbf F_k\mathbf P_{k-1}^+\mathbf F_k^T+\mathbf Q_k
       \]
    3. **测量更新**
       \[
       \mathbf K_k=\mathbf P_k^-\mathbf H_k^T
       (\mathbf H_k\mathbf P_k^-\mathbf H_k^T+\mathbf R_k)^{-1}
       \]
       \[
       \hat{\mathbf x}_k^+=\hat{\mathbf x}_k^-+
       \mathbf K_k(\mathbf z_k-\mathbf H_k\hat{\mathbf x}_k^-)
       \]

    ---

    ## 8. 总结

    卡尔曼滤波器维护两个对象：状态均值 $\hat{\mathbf x}$ 与协方差 $\mathbf P$。
    每一轮先用系统模型预测，再用传感器测量修正。卡尔曼增益不是手动设置的常数，
    而是由预测不确定性和测量不确定性共同决定。

    只要系统动力学和观测模型是线性的、噪声可以合理近似为高斯分布，上述递推就是
    线性最小均方误差意义下的最优估计。对于非线性系统，扩展卡尔曼滤波器（EKF）
    会在当前均值附近对非线性预测函数和观测函数进行局部线性化。

    ## 符号表

    | 符号 | 意义 |
    |---|---|
    | $\mathbf x_k$ | 时刻 $k$ 的真实状态向量 |
    | $p_k,\ v_k$ | 位置与速度 |
    | $\hat{\mathbf x}_k^-$ | 测量更新前的先验状态估计 |
    | $\hat{\mathbf x}_k^+$ | 测量更新后的后验状态估计 |
    | $\mathbf P_k^-,\ \mathbf P_k^+$ | 先验与后验协方差 |
    | $\mathbf F_k$ | 状态转移矩阵 |
    | $\mathbf B_k$ | 控制矩阵 |
    | $\mathbf u_k$ | 已知控制输入 |
    | $\mathbf Q_k$ | 过程噪声协方差 |
    | $\mathbf H_k$ | 观测矩阵 |
    | $\mathbf z_k$ | 传感器实际读数 |
    | $\mathbf R_k$ | 测量噪声协方差 |
    | $\mathbf y_k$ | 创新，即测量与预测读数之差 |
    | $\mathbf S_k$ | 创新协方差 |
    | $\mathbf K_k$ | 卡尔曼增益 |
    | $\Delta t$ | 相邻时刻之间的时间间隔 |
    | $\boldsymbol\mu,\mathbf\Sigma$ | 一般高斯分布的均值与协方差 |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 附录2 协方差矩阵的变换

    设随机向量 $x\in\mathbb{R}^n$，均值为

    $$
    \mu = \mathbb{E}[x]
    $$

    协方差矩阵定义为

    $$
    \Sigma = \operatorname{Cov}(x)
    = \mathbb{E}\left[(x-\mu)(x-\mu)^T\right]
    $$

    现在用一个矩阵 $\mathbf A\in\mathbb{R}^{m\times n}$ 对每个点做线性变换：

    $$
    y = \mathbf A x
    $$

    我们要求的是

    $$
    \operatorname{Cov}(y)
    =
    \operatorname{Cov}(\mathbf A x)
    $$

    ---

    ## 1. 先求变换后的均值

    因为 $\mathbf A$ 是常数矩阵，可以从期望中提出：

    $$
    \mathbb{E}[y]
    =
    \mathbb{E}[\mathbf A x]
    =
    \mathbf A\mathbb{E}[x]
    =
    \mathbf A\mu
    $$

    因此，变换后的均值是

    $$
    \mu_y = \mathbf A\mu
    $$

    ---

    ## 2. 从协方差定义出发

    根据协方差矩阵定义，

    $$
    \operatorname{Cov}(y)
    =
    \mathbb{E}\left[(y-\mu_y)(y-\mu_y)^T\right]
    $$

    代入

    $$
    y=\mathbf A x,
    \qquad
    \mu_y=\mathbf A\mu
    $$

    得到

    $$
    \operatorname{Cov}(\mathbf A x)
    =
    \mathbb{E}\left[
    (\mathbf A x-\mathbf A\mu)
    (\mathbf A x-\mathbf A\mu)^T
    \right]
    $$

    因为

    $$
    \mathbf A x-\mathbf A\mu
    =
    \mathbf A(x-\mu)
    $$

    所以

    $$
    \operatorname{Cov}(\mathbf A x)
    =
    \mathbb{E}\left[
    \mathbf A(x-\mu)
    \left(\mathbf A(x-\mu)\right)^T
    \right]
    $$

    ---

    ## 3. 处理转置

    使用矩阵转置规则：

    $$
    (BC)^T = C^T B^T
    $$

    因此

    $$
    \left(\mathbf A(x-\mu)\right)^T
    =
    (x-\mu)^T\mathbf A^T
    $$

    于是

    $$
    \operatorname{Cov}(\mathbf A x)
    =
    \mathbb{E}\left[
    \mathbf A(x-\mu)(x-\mu)^T\mathbf A^T
    \right]
    $$

    由于 $\mathbf A$ 和 $\mathbf A^T$ 都不是随机变量，而是常数矩阵，所以可以从期望中提出：

    $$
    \operatorname{Cov}(\mathbf A x)
    =
    \mathbf A
    \mathbb{E}\left[(x-\mu)(x-\mu)^T\right]
    \mathbf A^T
    $$

    又因为

    $$
    \mathbb{E}\left[(x-\mu)(x-\mu)^T\right]
    =
    \operatorname{Cov}(x)
    =
    \Sigma
    $$

    所以

    $$
    \boxed{
    \operatorname{Cov}(\mathbf A x)
    =
    \mathbf A\Sigma\mathbf A^T
    }
    $$

    这就是恒等式

    $$
    \operatorname{Cov}(\mathbf A x)
    =
    \mathbf A\operatorname{Cov}(x)\mathbf A^T
    $$

    的来源。

    ---

    ## 4. 仿射变换的情况

    如果不是单纯的线性变换，而是仿射变换

    $$
    y = \mathbf A x + b
    $$

    其中 $b$ 是常数向量，那么

    $$
    \mathbb{E}[y]
    =
    \mathbb{E}[\mathbf A x+b]
    =
    \mathbf A\mu+b
    $$

    协方差为

    $$
    \operatorname{Cov}(\mathbf A x+b)
    =
    \mathbb{E}
    \left[
    (\mathbf A x+b-\mathbf A\mu-b)
    (\mathbf A x+b-\mathbf A\mu-b)^T
    \right]
    $$

    中间的 $b$ 会抵消：

    $$
    \mathbf A x+b-\mathbf A\mu-b
    =
    \mathbf A(x-\mu)
    $$

    因此

    $$
    \operatorname{Cov}(\mathbf A x+b)
    =
    \mathbf A\Sigma\mathbf A^T
    $$

    也就是说，平移 $b$ 不会改变协方差，因为协方差只关心随机变量相对于均值的偏移。

    最终结论为

    $$
    \boxed{
    \operatorname{Cov}(\mathbf A x)
    =
    \mathbf A\operatorname{Cov}(x)\mathbf A^T
    =
    \mathbf A\Sigma\mathbf A^T
    }
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    # 附录3 高斯分布相乘得到高斯分布

    设两个 \(d\) 维高斯分布为

    \[
    p_1(x)=\mathcal N(x;\mu_1,\Sigma_1),
    \qquad
    p_2(x)=\mathcal N(x;\mu_2,\Sigma_2)
    \]

    其中

    \[
    x,\mu_1,\mu_2\in\mathbb R^d,
    \qquad
    \Sigma_1,\Sigma_2\in\mathbb R^{d\times d}
    \]

    且 \(\Sigma_1,\Sigma_2\) 对称正定。

    ## 1. 写出多维高斯密度

    \[
    p_1(x)
    =
    \frac{1}{(2\pi)^{d/2}|\Sigma_1|^{1/2}}
    \exp\left[
    -\frac12 (x-\mu_1)^T\Sigma_1^{-1}(x-\mu_1)
    \right]
    \]

    \[
    p_2(x)
    =
    \frac{1}{(2\pi)^{d/2}|\Sigma_2|^{1/2}}
    \exp\left[
    -\frac12 (x-\mu_2)^T\Sigma_2^{-1}(x-\mu_2)
    \right]
    \]

    令

    \[
    \Lambda_1=\Sigma_1^{-1},
    \qquad
    \Lambda_2=\Sigma_2^{-1}
    \]

    这里 \(\Lambda_1,\Lambda_2\) 称为**精度矩阵**。

    ## 2. 两个高斯相乘

    忽略与 \(x\) 无关的常数，有

    \[
    p_1(x)p_2(x)
    \propto
    \exp\left[
    -\frac12 (x-\mu_1)^T\Lambda_1(x-\mu_1)
    -\frac12 (x-\mu_2)^T\Lambda_2(x-\mu_2)
    \right]
    \]

    合并指数：

    \[
    p_1(x)p_2(x)
    \propto
    \exp\left[
    -\frac12
    \left(
    (x-\mu_1)^T\Lambda_1(x-\mu_1)
    +
    (x-\mu_2)^T\Lambda_2(x-\mu_2)
    \right)
    \right]
    \]

    ## 3. 展开二次型

    展开第一项：

    \[
    (x-\mu_1)^T\Lambda_1(x-\mu_1)
    =
    x^T\Lambda_1x
    -2\mu_1^T\Lambda_1x
    +
    \mu_1^T\Lambda_1\mu_1
    \]

    类似地，

    \[
    (x-\mu_2)^T\Lambda_2(x-\mu_2)
    =
    x^T\Lambda_2x
    -2\mu_2^T\Lambda_2x
    +
    \mu_2^T\Lambda_2\mu_2
    \]

    所以指数中的二次项为

    \[
    x^T(\Lambda_1+\Lambda_2)x
    -
    2(\Lambda_1\mu_1+\Lambda_2\mu_2)^T x
    +
    \mu_1^T\Lambda_1\mu_1
    +
    \mu_2^T\Lambda_2\mu_2
    \]

    因此

    \[
    p_1(x)p_2(x)
    \propto
    \exp\left[
    -\frac12
    \left(
    x^T(\Lambda_1+\Lambda_2)x
    -
    2(\Lambda_1\mu_1+\Lambda_2\mu_2)^Tx
    +
    \text{常数}
    \right)
    \right]
    \]

    其中最后的常数与 \(x\) 无关。

    ## 4. 配方成新的高斯形式

    希望将其写成

    \[
    -\frac12(x-\mu)^T\Sigma^{-1}(x-\mu)
    \]

    展开目标形式：

    \[
    (x-\mu)^T\Sigma^{-1}(x-\mu)
    =
    x^T\Sigma^{-1}x
    -
    2\mu^T\Sigma^{-1}x
    +
    \mu^T\Sigma^{-1}\mu
    \]

    对比前面的二次项：

    \[
    x^T(\Lambda_1+\Lambda_2)x
    -
    2(\Lambda_1\mu_1+\Lambda_2\mu_2)^Tx
    \]

    得到

    \[
    \Sigma^{-1}
    =
    \Lambda_1+\Lambda_2
    \]

    即

    \[
    \boxed{
    \Sigma
    =
    (\Lambda_1+\Lambda_2)^{-1}
    =
    (\Sigma_1^{-1}+\Sigma_2^{-1})^{-1}
    }
    \]

    再比较一次项：

    \[
    \Sigma^{-1}\mu
    =
    \Lambda_1\mu_1+\Lambda_2\mu_2
    \]

    因此

    \[
    \mu
    =
    \Sigma(\Lambda_1\mu_1+\Lambda_2\mu_2)
    \]

    代回 \(\Lambda_i=\Sigma_i^{-1}\)，得到

    \[
    \boxed{
    \mu
    =
    (\Sigma_1^{-1}+\Sigma_2^{-1})^{-1}
    (\Sigma_1^{-1}\mu_1+\Sigma_2^{-1}\mu_2)
    }
    \]

    ## 5. 结论

    因此

    \[
    \mathcal N(x;\mu_1,\Sigma_1)
    \mathcal N(x;\mu_2,\Sigma_2)
    \propto
    \mathcal N(x;\mu,\Sigma)
    \]

    其中

    \[
    \boxed{
    \Sigma
    =
    (\Sigma_1^{-1}+\Sigma_2^{-1})^{-1}
    }
    \]

    \[
    \boxed{
    \mu
    =
    \Sigma(\Sigma_1^{-1}\mu_1+\Sigma_2^{-1}\mu_2)
    }
    \]

    也就是说，任意维度下，两个关于同一个变量 \(x\) 的高斯分布相乘，仍然正比于一个高斯分布。
    """)
    return


if __name__ == "__main__":
    app.run()
