from dataclasses import dataclass, field


@dataclass
class MemoryPlan:
    slots: list = field(default_factory=list)
    assignments: dict = field(default_factory=dict)
    last_use: dict = field(default_factory=dict)
    events: list = field(default_factory=list)


def plan_memory(scheduler):
    """Reuse only dead intermediate storage, never an input or returned output."""
    plan = MemoryPlan()
    graph = scheduler.graph
    materialized = [scheduler.materialized_outputs(node) for node in scheduler.nodes]
    for i, node in enumerate(scheduler.nodes):
        for name in sorted(set(materialized[i]) | node.reads):
            plan.last_use[name] = i
    for name in graph.output_names:
        plan.last_use[name] = len(scheduler.nodes)

    free = []
    for i, names in enumerate(materialized):
        for name in names:
            layout = graph.buffers[name].layout
            slot = next((slot for slot in free if plan.slots[slot] == layout), None)
            action = "reuse"
            if slot is None:
                action = "allocate"
                slot = len(plan.slots)
                plan.slots.append(layout)
            else:
                free.remove(slot)
            plan.assignments[name] = slot
            plan.events.append({"kernel": i, "action": action, "buffer": name, "slot": slot})

        # Release after the launch, so its reads cannot alias newly allocated writes.
        for name, slot in plan.assignments.items():
            if plan.last_use[name] == i:
                free.append(slot)
                plan.events.append({"kernel": i, "action": "release", "buffer": name, "slot": slot})
    return plan
