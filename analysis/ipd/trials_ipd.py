"""Registry of digitized KM inputs for pseudo-IPD reconstruction.

One entry per trial -> endpoint -> arm. Each arm gives:
  coords : CSV in digitized/ with columns time,surv (survival 0-1), read off the
           published/slide KM curve and anchored to reported landmark survivals.
  nar_t / nar : the number-at-risk table (months / counts) read verbatim.
  n        : arm total (= number at risk at t=0).
Each endpoint also stores the published HR (point, lo, hi) for validation and a
tau (months) for RMST. `source` flags published (peer-reviewed) vs slide (congress)
— slide trials are the sensitivity layer, published trials the primary IPD set.

Arm convention for Cox: exposure arm = WPRT/pelvic RT coded 1, PORT coded 0, so
HR<1 favours WPRT (consistent with the aggregate analysis).
"""

TRIALS_IPD = {
    'POP-RT': {
        'source': 'published',
        'endpoints': {
            'BFFS': {
                'published_hr': (0.23, 0.10, 0.52),
                'tau': 96,
                'arms': {
                    'WPRT': {'n': 110, 'coords': 'POPRT_BFFS_WPRT.csv',
                             'nar_t': [0, 12, 24, 36, 48, 60, 72, 84, 96],
                             'nar': [110, 106, 104, 100, 81, 64, 40, 20, 10]},
                    'PORT': {'n': 112, 'coords': 'POPRT_BFFS_PORT.csv',
                             'nar_t': [0, 12, 24, 36, 48, 60, 72, 84, 96],
                             'nar': [112, 106, 104, 97, 77, 55, 34, 22, 10]},
                },
            },
            'DMFS': {
                'published_hr': (0.35, 0.15, 0.82),
                'tau': 96,
                'arms': {
                    'WPRT': {'n': 110, 'coords': 'POPRT_DMFS_WPRT.csv',
                             'nar_t': [0, 12, 24, 36, 48, 60, 72, 84, 96],
                             'nar': [110, 107, 105, 100, 80, 64, 41, 20, 9]},
                    'PORT': {'n': 112, 'coords': 'POPRT_DMFS_PORT.csv',
                             'nar_t': [0, 12, 24, 36, 48, 60, 72, 84, 96],
                             'nar': [112, 108, 107, 99, 80, 56, 39, 21, 10]},
                },
            },
            'OS': {
                'published_hr': (0.92, 0.41, 2.05),
                'tau': 96,
                'arms': {
                    'WPRT': {'n': 110, 'coords': 'POPRT_OS_WPRT.csv',
                             'nar_t': [0, 12, 24, 36, 48, 60, 72, 84, 96],
                             'nar': [110, 108, 105, 101, 83, 67, 43, 21, 10]},
                    'PORT': {'n': 112, 'coords': 'POPRT_OS_PORT.csv',
                             'nar_t': [0, 12, 24, 36, 48, 60, 72, 84, 96],
                             'nar': [112, 110, 108, 102, 85, 65, 47, 29, 15]},
                },
            },
        },
    },

    'GETUG-01': {
        'source': 'published',
        'endpoints': {
            # Long-term update, whole population (panel A). x-axis in years -> months.
            'EFS': {  # event-free survival = biochemical/progression endpoint
                'published_hr': (1.05, 0.83, 1.33),
                'tau': 120,
                'arms': {
                    'WPRT': {'n': 175, 'coords': 'GETUG01_EFS_WPRT.csv',
                             'nar_t': [0, 24, 48, 72, 96, 120, 144, 168, 192],
                             'nar': [175, 144, 102, 70, 60, 42, 18, 5, 0]},
                    'PORT': {'n': 175, 'coords': 'GETUG01_EFS_PORT.csv',
                             'nar_t': [0, 24, 48, 72, 96, 120, 144, 168, 192],
                             'nar': [175, 150, 110, 83, 59, 36, 15, 5, 0]},
                },
            },
            'OS': {
                'published_hr': (0.88, 0.63, 1.22),
                'tau': 120,
                'arms': {
                    'WPRT': {'n': 177, 'coords': 'GETUG01_OS_WPRT.csv',
                             'nar_t': [0, 24, 48, 72, 96, 120, 144, 168, 192],
                             'nar': [177, 166, 141, 120, 106, 79, 40, 14, 0]},
                    'PORT': {'n': 176, 'coords': 'GETUG01_OS_PORT.csv',
                             'nar_t': [0, 24, 48, 72, 96, 120, 144, 168, 192],
                             'nar': [176, 164, 139, 124, 104, 76, 35, 11, 0]},
                },
            },
        },
    },

    # ---- Sensitivity layer: not-yet-peer-reviewed (congress slides) ----
    'PEACE-2': {
        'source': 'slide',  # ESTRO 2026 pelvic-RT randomization (NOT the cabazitaxel arms)
        'endpoints': {
            'BPFS': {  # biochemical progression-free survival
                'published_hr': (0.84, 0.66, 1.07),
                'tau': 96,
                'arms': {
                    'WPRT': {'n': 381, 'coords': 'PEACE2_BPFS_WPRT.csv',
                             'nar_t': [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120],
                             'nar': [381, 360, 347, 328, 295, 225, 163, 115, 71, 42, 17]},
                    'PORT': {'n': 380, 'coords': 'PEACE2_BPFS_PORT.csv',
                             'nar_t': [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120],
                             'nar': [380, 360, 344, 326, 286, 225, 153, 113, 79, 46, 16]},
                },
            },
            'MFS': {
                'published_hr': (0.90, 0.69, 1.19),
                'tau': 96,
                'arms': {
                    'WPRT': {'n': 381, 'coords': 'PEACE2_MFS_WPRT.csv',
                             'nar_t': [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120],
                             'nar': [381, 365, 352, 334, 315, 259, 198, 148, 104, 67, 30]},
                    'PORT': {'n': 380, 'coords': 'PEACE2_MFS_PORT.csv',
                             'nar_t': [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120],
                             'nar': [380, 364, 349, 333, 313, 258, 199, 154, 110, 69, 30]},
                },
            },
            'OS': {
                'published_hr': (1.21, 0.84, 1.73),
                'tau': 96,
                'arms': {
                    'WPRT': {'n': 381, 'coords': 'PEACE2_OS_WPRT.csv',
                             'nar_t': [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120],
                             'nar': [381, 367, 361, 355, 344, 293, 229, 172, 124, 79, 35]},
                    'PORT': {'n': 380, 'coords': 'PEACE2_OS_PORT.csv',
                             'nar_t': [0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120],
                             'nar': [380, 366, 364, 358, 346, 293, 238, 183, 130, 89, 42]},
                },
            },
        },
    },

    'RTOG 0924': {
        'source': 'slide',  # ASTRO 2025 late-breaking. Only OS has a digitizable KM
        'endpoints': {      # (DM/PCSM/biochemical shown as competing-risks CIF -> not reconstructed)
            'OS': {
                'published_hr': (1.01, 0.85, 1.20),
                'tau': 120,
                'arms': {
                    'WPRT': {'n': 1245, 'coords': 'RTOG0924_OS_WPRT.csv',
                             'nar_t': [0, 36, 72, 108, 144],
                             'nar': [1245, 1107, 819, 246, 30]},
                    'PORT': {'n': 1228, 'coords': 'RTOG0924_OS_PORT.csv',
                             'nar_t': [0, 36, 72, 108, 144],
                             'nar': [1228, 1116, 825, 255, 34]},
                },
            },
        },
    },
}
