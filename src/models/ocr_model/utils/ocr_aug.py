from augraphy import *
import random

def ocr_augmentation_pipeline():
    pre_phase = [
        # Rescale(scale="optimal", target_dpi = 300,  p = 1.0),
    ]

    ink_phase = [
        # 6ms
        InkColorSwap(
            ink_swap_color="random",
            ink_swap_sequence_number_range=(5, 10),
            ink_swap_min_width_range=(2, 3),
            ink_swap_max_width_range=(100, 120),
            ink_swap_min_height_range=(2, 3),
            ink_swap_max_height_range=(100, 120),
            ink_swap_min_area_range=(10, 20),
            ink_swap_max_area_range=(400, 500),
            p=0.1
        ),
        # 10ms
        Dithering(
            dither=random.choice(["ordered", "floyd-steinberg"]),
            order=(3, 5),
            p=0.05
        ),
        # 10ms
        InkBleed(
            intensity_range=(0.1, 0.2),
            kernel_size=random.choice([(7, 7), (5, 5), (3, 3)]),
            severity=(0.4, 0.6),
            p=0.2,
        ),
        # 40ms
        InkShifter(
            text_shift_scale_range=(18, 27),
            text_shift_factor_range=(1, 4),
            text_fade_range=(0, 2),
            blur_kernel_size=(5, 5),
            blur_sigma=0,
            noise_type="random",
            p=0.1
        ),
        # 90ms
        # Letterpress(
        #     n_samples=(100, 400),
        #     n_clusters=(200, 400),
        #     std_range=(500, 3000),
        #     value_range=(150, 224),
        #     value_threshold_range=(96, 128),
        #     blur=1,
        #     p=0.1
        # ),
    ]

    paper_phase = [
        # 50ms
        # OneOf(
        #     [
        #         ColorPaper(
        #             hue_range=(0, 255),
        #             saturation_range=(10, 40),
        #         ),
        #         PatternGenerator(
        #             imgx=random.randint(256, 512),
        #             imgy=random.randint(256, 512),
        #             n_rotation_range=(10, 15),
        #             color="random",
        #             alpha_range=(0.25, 0.5),
        #         ),
        #         NoiseTexturize(
        #             sigma_range=(3, 10),
        #             turbulence_range=(2, 5),
        #             texture_width_range=(300, 500),
        #             texture_height_range=(300, 500),
        #         ),
        #     ],
        #     p=0.05
        # ),
        # 10ms
        BrightnessTexturize(
            texturize_range=(0.9, 0.99),
            deviation=0.03,
            p=0.1
        )
    ]

    post_phase = [
        # 13ms
        ColorShift(
            color_shift_offset_x_range=(3, 5),
            color_shift_offset_y_range=(3, 5),
            color_shift_iterations=(2, 3),
            color_shift_brightness_range=(0.9, 1.1),
            color_shift_gaussian_kernel_range=(3, 3),
            p=0.05
        ),
        # 13ms
        DirtyDrum(
            line_width_range=(1, 6),
            line_concentration=random.uniform(0.05, 0.15),
            direction=random.randint(0, 2),
            noise_intensity=random.uniform(0.6, 0.95),
            noise_value=(64, 224),
            ksize=random.choice([(3, 3), (5, 5), (7, 7)]),
            sigmaX=0,
            p=0.05,
        ),
        # 10ms
        OneOf(
            [
                LightingGradient(
                    light_position=None,
                    direction=None,
                    max_brightness=255,
                    min_brightness=0,
                    mode="gaussian",
                    linear_decay_rate=None,
                    transparency=None,
                ),
                Brightness(
                    brightness_range=(0.9, 1.1),
                    min_brightness=0,
                    min_brightness_value=(120, 150),
                ),
                Gamma(
                    gamma_range=(0.9, 1.1),
                ),
            ],
            p=0.05
        ),
        # 6ms
        Jpeg(
            quality_range=(25, 95),
            p=0.1
        ),
        # 12ms
        Markup(
            num_lines_range=(2, 7),
            markup_length_range=(0.5, 1),
            markup_thickness_range=(1, 2),
            markup_type=random.choice(["strikethrough", "crossed", "highlight", "underline"]),
            markup_color="random",
            single_word_mode=False,
            repetitions=1,
            p=0.05
        ),
        # 65ms
        # OneOf(
        #     [
        #         BadPhotoCopy(
        #             noise_mask=None,
        #             noise_type=-1,
        #             noise_side="random",
        #             noise_iteration=(1, 2),
        #             noise_size=(1, 3),
        #             noise_value=(128, 196),
        #             noise_sparsity=(0.3, 0.6),
        #             noise_concentration=(0.1, 0.6),
        #             blur_noise=random.choice([True, False]),
        #             blur_noise_kernel=random.choice([(3, 3), (5, 5), (7, 7)]),
        #             wave_pattern=random.choice([True, False]),
        #             edge_effect=random.choice([True, False]),
        #         ),
        #         ShadowCast(
        #             shadow_side="random",
        #             shadow_vertices_range=(1, 20),
        #             shadow_width_range=(0.3, 0.8),
        #             shadow_height_range=(0.3, 0.8),
        #             shadow_color=(0, 0, 0),
        #             shadow_opacity_range=(0.2, 0.9),
        #             shadow_iterations_range=(1, 2),
        #             shadow_blur_kernel_range=(101, 301),
        #         ),
        #         LowLightNoise(
        #             num_photons_range=(50, 100),
        #             alpha_range=(0.7, 1.0),
        #             beta_range=(10, 30),
        #             gamma_range=(1, 1.8),
        #             bias_range=(20, 40),
        #             dark_current_value=1.0,
        #             exposure_time=0.2,
        #             gain=0.1,
        #         ),
        #     ],
        #     p=0.05,
        # ),
        # 10ms
        OneOf(
            [
                NoisyLines(
                    noisy_lines_direction="random",
                    noisy_lines_location="random",
                    noisy_lines_number_range=(5, 20),
                    noisy_lines_color=(0, 0, 0),
                    noisy_lines_thickness_range=(1, 2),
                    noisy_lines_random_noise_intensity_range=(0.01, 0.1),
                    noisy_lines_length_interval_range=(0, 100),
                    noisy_lines_gaussian_kernel_value_range=(3, 5),
                    noisy_lines_overlay_method="ink_to_paper",
                ),
                BindingsAndFasteners(
                    overlay_types="darken",
                    foreground=None,
                    effect_type="random",
                    width_range="random",
                    height_range="random",
                    angle_range=(-30, 30),
                    ntimes=(2, 6),
                    nscales=(0.9, 1.0),
                    edge="random",
                    edge_offset=(10, 50),
                    use_figshare_library=0,
                ),
            ],
            p=0.05,
        ),
        # 20ms
        OneOf(
            [
                PageBorder(
                    page_border_width_height="random",
                    page_border_color=(0, 0, 0),
                    page_border_background_color=(0, 0, 0),
                    page_numbers="random",
                    page_rotation_angle_range=(-3, 3),
                    curve_frequency=(2, 8),
                    curve_height=(2, 4),
                    curve_length_one_side=(50, 100),
                    same_page_border=random.choice([0, 1]),
                ),
                Folding(
                    fold_x=None,
                    fold_deviation=(0, 0),
                    fold_count=random.randint(2, 8),
                    fold_noise=0.01,
                    fold_angle_range=(-360, 360),
                    gradient_width=(0.1, 0.2),
                    gradient_height=(0.01, 0.02),
                    backdrop_color=(0, 0, 0),
                ),
            ],
            p=0.05
        ),
    ]

    pipeline = AugraphyPipeline(
        ink_phase=ink_phase,
        paper_phase=paper_phase,
        post_phase=post_phase,
        pre_phase=pre_phase,
        log=False,
    )

    return pipeline