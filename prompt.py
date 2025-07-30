TASK_DEFINITION = """
You are a RL(PPO) decision-making policy/value network for an autonomous vehicle.
Your task is to choose the **next driving action** that balances both **safety** and **efficiency**. 
- Safety means avoiding collisions and maintaining safe distances between other vehicles.
- Efficiency means maintaining a reasonable speed and making progress toward the goal.
"""

TRAFFIC_PREFERENCE = """
1. Try to keep a safe distance to the car in front of you.
2. Try to gain efficiency through active lane change or acceleration.
"""


DECISION_CAUTIONS = """
1. You must output a decision when you finish this task. 
2. Your final output decision must be unique and not ambiguous.
"""

ACTIONS_DESCRIPTION = """
The following actions are available for the ego vehicle:

Action 0: LANE_LEFT
Description: Change lane to the left of the current lane.

Action 1: IDLE
Description: Remain in the current lane.

Action 2: LANE_RIGHT
Description: Change lane to the right of the current lane.

Please select an action for ego vehicle based on the corresponding current scenario.
"""



def build_prompt(current_scenario):
    message_prefix = TASK_DEFINITION
    traffic_rules = TRAFFIC_PREFERENCE
    decision_cautions = DECISION_CAUTIONS
    action_space = ACTIONS_DESCRIPTION

    prompt = (f"{message_prefix}"
              "There are several rules you need to follow when you drive on a highway:\n"
              f"{traffic_rules}\n\n"
              "Here are your attention points:\n"
              f"{decision_cautions}\n\n"
              f"{action_space}\n\n")
              
    user_prompt = ("Below is the information for the current scenario:\n" 
                    f"{current_scenario}\n\n")
    return prompt + user_prompt


    
def build_single_scenario_prompt(observation):
    if len(observation) == 0:
        return "No vehicles detected in the scene."

    def get_lane(y):
        if 10 <= y < 14:
            return +3
        elif 6 <= y < 10:
            return +2
        elif 2 <= y < 6:
            return +1
        elif -2 <= y < 2:
            return 0
        elif -6 <= y < -2:
            return -1
        elif -10 <= y < -6:
            return -2
        elif -14 <= y < -10:
            return -3
        else:
            return None

    ego = observation[0]
    _, ego_x, ego_y, ego_vx, ego_vy = ego
    ego_lane = get_lane(ego_y)

    def relative_position(target_lane):
        if target_lane is None or ego_lane is None:
            return "unknown"
        offset = target_lane - ego_lane
        if offset == 0:
            return "same lane as ego"
        elif offset > 0:
            return f"{abs(offset)} lane(s) to the right of ego"
        else:
            return f"{abs(offset)} lane(s) to the left of ego"

    ego_info = (
        f"Ego Vehicle:\n"
        f"- Position (x, y): ({ego_x:.2f}, {ego_y:.2f})\n"
        f"- Speed (vx, vy): ({ego_vx:.2f}, {ego_vy:.2f})\n"
        f"- Lane Index: {ego_lane}\n"
    )

    others_info = []
    for i, vehicle in enumerate(observation[1:], start=1):
        presence, x, y, vx, vy = vehicle
        if presence == 1:
            lane = get_lane(y)
            rel_pos = relative_position(lane)
            dx = x - ego_x
            dist_str = f"{abs(dx):.1f} m ahead" if dx > 0 else f"{abs(dx):.1f} m behind"
            others_info.append(
                f"Vehicle {i}:\n"
                f"- Position (x, y): ({x:.2f}, {y:.2f})\n"
                f"- Speed (vx, vy): ({vx:.2f}, {vy:.2f})\n"
                f"- Relative Lane: {rel_pos}\n"
                f"- Relative Position: {dist_str}"
            )

    others_info_str = "\n\n".join(others_info) if others_info else "No other vehicles present."

    prompt = f"""
    Ego Vehicle State:
    {ego_info}

    Surrounding Vehicles:
    {others_info_str}
    """

    return prompt
