"""Asset -> (split, class) for the 50/25/25 four-class benchmark.

Single source of truth for all published numbers; matches
`assets/splits/train-test-val_split.yaml`.

`"val"` (21 assets, 6/7/4/4) selects checkpoints; `"test"` (24 assets, 8/8/4/4) is the
held-out split reported in the paper.
"""
from __future__ import annotations

DISTANCE_BINS: tuple[str, ...] = (
    "2r", "4r", "8r", "16r", "32r", "64r", "100r", "140r", "200r", "280r",
)
ELEVATIONS: tuple[str, ...] = ("20el", "-20el")
SWEEP_TAIL = "_45az_4cams_lgm_pinhole_3.0x_"
FRAMES_PER_BIN = 30

ASSIGNMENTS: dict[str, dict[int, list[str]]] = {
    # ~50%, fitting; every morphological bucket represented
    "train": {
        0: [  # birds (15): raptor, owl, corvid, pigeon, parrot, tropical, mythical, waterbird, misc
            "bald-eagle-med-poly",
            "jin-diao-vulture-kung-fu-panda-chi-master",
            "20_Owl_merged",
            "crow",
            "Crow_-_Hello_Neighbor_2",
            "Paloma_animada",
            "animated-bird-pigeon",
            "parrot",
            "animated-stylized-parrot-3d-animal-model",
            "animated-stylized-tukan-3d-animal-model",
            "phoenix",
            "Seagull",
            "canada-goose-in-flight",
            "KingFisher",
            "realistic-animated-woodpecker-3d-model",
        ],
        1: [  # drones (15): DJI, generic quad, FPV, race, hexa, octo, VTOL
            "Dji_Mavic_3_Classic_Drone",
            "dji_phantom_3",
            "DJI_Inspire_2",
            "quadcopter-dji-matrice-300-rtk",
            "dji_drone_dji_drone",
            "quadcopter_1",
            "Quadcopter_Drone_by_Atulya_Vaibhav_Pandey",
            "X4_Quad",
            "FPV-drone",
            "Race_Drone_-_Hunter",
            "AGRICULTURE_HEXACOPTER_DRONE",
            "drone-m600",
            "hexacopter-15",
            "DRONEE",
            "7Kg_VTOL_UAV",
        ],
        2: [  # airplanes (8): light prop, aerobatic, ag, COIN twin-prop, 3 airliners, military jet
            "Aerobatic-Airplane",
            "Cesna-Airplane",
            "Air_Tractor_AT-502",
            "FMA_IA_58_Pucara",
            "Airplane_A380",
            "Boeing_747-100",
            "Airplane_tomations",
            "Soko_G-4_Super_Galeb",
        ],
        3: [  # helicopters (8): 2 utility, light obs, stylised civil, attack, military, heavy/SAR, vintage
            "EC_135",
            "Helicopter_irs1182",
            "MD-500_Defender_Helicopter",
            "Helicopter_surveytown",
            "Attack-Helicopter",
            "Army_Helicopter",
            "Airbus-Helicopter-S365-Dauphin",
            "Bell_Huey_helicopter",
        ],
    },

    # ~25%, checkpoint selection; one per bucket
    "val": {
        0: [  # birds (7)
            "animated_eagle",
            "realistic-animated-owl-3d-model",
            "crow-fly",
            # "british_show_racer_pigeon",
            "animated-stylized-pigeon-3d-animal-model",
            "Quetzal_Animation",
            "realistic-animated-seagull-3d-model",
        ],
        1: [  # drones (7)
            "DJI_Mini_3_Pro",
            "Quadcopter_Assembly",
            "Race_Drone_-_Scout",
            "Drona_5_inch",
            "fpv-drone-without-a",
            "hexa-copter-4_snapshot_6",
            "VERSI_NUHINA_HINU_HYA",
        ],
        2: [  # airplanes (4)
            "Vintage_Toy_Airplane",
            "Airport_Airplane",
            "Grumman_F4F_Wildcat_Airplane",
            "Airplane_CRJ-900_Cityjet",
        ],
        3: [  # helicopters (4)
            "Helicopter_EC135_Special_Forces",
            "MH-6_Little_Bird_Helicopter_Animated",
            "Merlin_MK2_Helicopter",
            "Animated-civilian-Helicopter",
        ],
    },

    # ~25%, held-out eval; distinctive / hard cases
    "test": {
        0: [  # birds (8): incl. owl, crane, macaw, rare tropicals
            "realistic-animated-eagle-3d-model",
            "white-eagle-animation-fast-fly",
            "owl",
            "dove-bird-rigged",
            "pigeon-white",
            "cacau",
            "Silver_Ear_Bird",
            "Grus-nigricollis",
        ],
        1: [  # drones (8): incl. Tarot-650 hexa, low-poly, monobody, Inspire 3, asymmetric
            "DJI_Inspire_3",
            "AltAir_with_additional_rotor_arm_bearings_quadcopter",
            "Quadcopter_Drone_MonoBody",
            "quadcoptor_drone_assymbly",
            "Drone_v12",
            "ZD1250",
            "Low-poly_FPV",
            "Tarot-650",
        ],
        2: [  # airplanes (4): cargo prop, low-poly, vintage liner, biplane
            "cargo-aircraft",
            "BOEING-STEARMAN_MODEL_75",
            "Low-poly-Airplane",
            "1956_L-1049G_Super_Constellation_Full_Interior",
        ],
        3: [  # helicopters (4): helecopter, NOTAR Explorer, Dolphin, military
            "helecopter",
            "MD_Helicopters_MD-902_Explorer",
            "Dolphin_Helicopter__AS-365_Harbin_Z-9_",
            "Helicopter_Military",
        ],
    },
}
