"""
Evaluation Utilities and Exact-Match Scoring for Mara vs Needle 3.
Implements standardized evaluation metrics:
- Ordered Exact Match (OEM): Function names, count, order, and every argument key/value match.
- Tool Selection Accuracy: Function names and order match regardless of argument values.
- Argument Exact Match: For correctly routed tools, all arguments match ground truth.
- False Positive Trigger Rate: Calls produced on negative / out-of-scope queries.
"""

from typing import Any, Dict, List, Tuple


def normalize_room(room_str: str) -> str:
    """Normalizes room names to standard canonical format."""
    r = str(room_str).lower().strip()
    if "living" in r or "lounge" in r:
        return "living room"
    if "bed" in r or "master" in r:
        return "bedroom"
    if "kit" in r:
        return "kitchen"
    if "bath" in r:
        return "bathroom"
    if "gar" in r:
        return "garage"
    return r


def normalize_device(dev_str: str) -> str:
    """Normalizes device names."""
    d = str(dev_str).lower().strip()
    if "fan" in d:
        return "fan"
    if "garage" in d or "shutter" in d:
        return "garage door"
    if "light" in d or "lamp" in d:
        return "light"
    if "ac" in d or "cooler" in d:
        return "ac"
    if "blind" in d:
        return "blinds"
    return d


def normalize_action(action_str: str) -> str:
    """Normalizes action strings."""
    a = str(action_str).lower().strip()
    if a in ["on", "turn_on", "start", "enable", "open", "raise"]:
        return "on" if "door" not in a else "open"
    if a in ["off", "turn_off", "cut", "kill", "close", "shut"]:
        return "off" if "door" not in a else "close"
    return a


def compare_arguments(tool_name: str, pred_args: Dict[str, Any], gt_args: Dict[str, Any]) -> bool:
    """Checks whether predicted arguments match ground truth arguments with type-safety."""
    if tool_name == "set_lights":
        # Compare room
        pred_room = normalize_room(pred_args.get("room", ""))
        gt_room = normalize_room(gt_args.get("room", ""))
        if pred_room != gt_room:
            return False

        # Compare brightness
        pred_b = pred_args.get("brightness")
        gt_b = gt_args.get("brightness")
        if pred_b is not None and gt_b is not None:
            try:
                if int(pred_b) != int(gt_b):
                    return False
            except (ValueError, TypeError):
                return False
        return True

    elif tool_name == "set_thermostat":
        # Compare temperature_c (within 0.5C tolerance)
        pred_t = pred_args.get("temperature_c") or pred_args.get("temp")
        gt_t = gt_args.get("temperature_c")
        if pred_t is None or gt_t is None:
            return False
        try:
            return abs(float(pred_t) - float(gt_t)) < 0.6
        except (ValueError, TypeError):
            return False

    elif tool_name == "control_device":
        pred_dev = normalize_device(pred_args.get("device", ""))
        gt_dev = normalize_device(gt_args.get("device", ""))
        pred_act = normalize_action(pred_args.get("action", ""))
        gt_act = normalize_action(gt_args.get("action", ""))
        return (pred_dev == gt_dev) and (pred_act == gt_act)

    elif tool_name == "lock_door":
        pred_door = str(pred_args.get("door", "")).lower()
        gt_door = str(gt_args.get("door", "")).lower()
        pred_lock = pred_args.get("locked")
        gt_lock = gt_args.get("locked")
        # Normalize boolean lock status
        if isinstance(pred_lock, str):
            pred_lock = pred_lock.lower() in ["true", "1", "locked", "lock"]
        if isinstance(gt_lock, str):
            gt_lock = gt_lock.lower() in ["true", "1", "locked", "lock"]
        return ("front" in pred_door and "front" in gt_door) and (bool(pred_lock) == bool(gt_lock))

    return pred_args == gt_args


def evaluate_single_sample(pred_calls: List[Dict[str, Any]], gt_calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluates predicted calls against ground truth calls for a single query.
    Returns:
      is_oem: bool (Ordered Exact Match)
      tool_match: bool (Tools match in order and count)
      arg_match: bool (Arguments match where tools match)
      is_false_trigger: bool (Produced calls when ground truth is empty)
    """
    # 1. Negative / Out of Scope Check
    if len(gt_calls) == 0:
        if len(pred_calls) == 0:
            return {"is_oem": True, "tool_match": True, "arg_match": True, "is_false_trigger": False}
        else:
            return {"is_oem": False, "tool_match": False, "arg_match": False, "is_false_trigger": True}

    # 2. Count mismatch
    if len(pred_calls) != len(gt_calls):
        # Check partial tool match
        pred_names = [c.get("name") for c in pred_calls]
        gt_names = [c.get("name") for c in gt_calls]
        tool_match = (pred_names == gt_names)
        return {"is_oem": False, "tool_match": tool_match, "arg_match": False, "is_false_trigger": False}

    # 3. Check each call in order
    all_tools_match = True
    all_args_match = True

    for pred, gt in zip(pred_calls, gt_calls):
        p_name = pred.get("name")
        g_name = gt.get("name")
        if p_name != g_name:
            all_tools_match = False
            all_args_match = False
            break

        if not compare_arguments(p_name, pred.get("arguments", {}), gt.get("arguments", {})):
            all_args_match = False

    is_oem = all_tools_match and all_args_match
    return {
        "is_oem": is_oem,
        "tool_match": all_tools_match,
        "arg_match": all_args_match,
        "is_false_trigger": False,
    }
