"""Scene configuration registry."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SceneConfig:
    """Static description of a manipulation scene.

    objects:          list of (id, label, (x, y, z)) — initial positions used at first spawn.
    place_targets:    {source_label: default_target_label} hints for scripted eval.
    use_table:        when True, IsaacEnv adds a SeattleLabTable.
    random_spawn:     when True, IsaacEnv randomises (x, y) on every reset() call.
    spawn_bounds:     (x_min, y_min, x_max, y_max) workspace for random spawn.
    min_object_dist:  minimum centre-to-centre distance between spawned objects (m).
    """

    name: str
    objects: list[tuple[str, str, tuple[float, float, float]]]
    place_targets: dict[str, str] = field(default_factory=dict)
    use_table: bool = False
    random_spawn: bool = False
    spawn_bounds: tuple[float, float, float, float] = (0.38, -0.22, 0.62, 0.22)
    min_object_dist: float = 0.12
    object_size: float = 0.042   # cube side length (m); drives CuboidCfg and grasp depth


SCENES: dict[str, SceneConfig] = {
    # ------------------------------------------------------------------
    # tabletop_rb — two solid-colour boxes on a SeattleLabTable.
    # Positions are randomised on every reset (seed controlled by run_sim.py).
    # Panda reach: 0.855 m → workspace x 0.38–0.62, y ±0.22 fits comfortably.
    "tabletop_rb": SceneConfig(
        name="tabletop_rb",
        objects=[
            ("red_cube",  "red_cube",  (0.50, -0.10, 0.00)),
            ("blue_cube", "blue_cube", (0.50,  0.10, 0.00)),
        ],
        place_targets={"red_cube": "blue_cube"},
        use_table=True,
        random_spawn=True,
        spawn_bounds=(0.38, -0.22, 0.62, 0.22),
        min_object_dist=0.12,
        object_size=0.042,
    ),
}


def get_scene(name: str) -> SceneConfig:
    """Return SceneConfig by name.  Raises KeyError with available names on miss."""
    if name not in SCENES:
        raise KeyError(
            f"Unknown scene {name!r}. Available: {list(SCENES)}"
        )
    return SCENES[name]
