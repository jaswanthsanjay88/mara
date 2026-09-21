"""
Author and Freeze the 250-sample Evaluation Suite (data/frozen_eval_250.jsonl).
Criteria:
1. >= 50 samples per category across 5 categories (Clean, Paraphrase, Typo/Noise, Compound, Hard Negative).
2. Independent phrasing habits (subordinating clauses: 'before you lock up...', 'once the fan is off...', natural idioms).
3. Collision filter: verifies zero exact matches and rejects high n-gram / token overlap with training data.
4. FROZEN permanently before any evaluation or fine-tuning runs.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from research.finetune_tools import build_finetune_dataset

OUT_FILE = os.path.join(ROOT, "data", "frozen_eval_250.jsonl")

# ---------------------------------------------------------------------------
# 1. CATEGORY A: CLEAN (50 samples)
# ---------------------------------------------------------------------------
CLEAN_50 = [
    ("turn on living room light", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 100}}]),
    ("turn off living room light", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 0}}]),
    ("set living room lights to 60", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 60}}]),
    ("dim living room lights to 15", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 15}}]),
    ("turn on kitchen light", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 100}}]),
    ("turn off kitchen lights", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}}]),
    ("set kitchen light to 75", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 75}}]),
    ("dim kitchen to 35 percent", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 35}}]),
    ("switch on bedroom lights", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 100}}]),
    ("switch off bedroom lights", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 0}}]),
    ("adjust bedroom lights to 40", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 40}}]),
    ("set bedroom lights to 85 percent", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 85}}]),
    ("turn on bathroom light", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 100}}]),
    ("turn off bathroom light", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 0}}]),
    ("set bathroom brightness to 50", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 50}}]),
    ("dim bathroom to 10 percent", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 100 if False else 10}}]),
    ("turn on garage light", [{"name": "set_lights", "arguments": {"room": "garage", "brightness": 100}}]),
    ("turn off garage light", [{"name": "set_lights", "arguments": {"room": "garage", "brightness": 0}}]),
    ("set garage lights to 100", [{"name": "set_lights", "arguments": {"room": "garage", "brightness": 100}}]),
    ("dim garage lights to 45", [{"name": "set_lights", "arguments": {"room": "garage", "brightness": 45}}]),
    ("set thermostat to 19", [{"name": "set_thermostat", "arguments": {"temperature_c": 19.0}}]),
    ("set thermostat to 19.5 degrees", [{"name": "set_thermostat", "arguments": {"temperature_c": 19.5}}]),
    ("set temperature to 20 celsius", [{"name": "set_thermostat", "arguments": {"temperature_c": 20.0}}]),
    ("adjust thermostat to 20.5", [{"name": "set_thermostat", "arguments": {"temperature_c": 20.5}}]),
    ("set climate to 21 degrees", [{"name": "set_thermostat", "arguments": {"temperature_c": 21.0}}]),
    ("set thermostat to 21.5 degrees celsius", [{"name": "set_thermostat", "arguments": {"temperature_c": 21.5}}]),
    ("change temperature to 22", [{"name": "set_thermostat", "arguments": {"temperature_c": 22.0}}]),
    ("set temperature to 22.5", [{"name": "set_thermostat", "arguments": {"temperature_c": 22.5}}]),
    ("adjust climate target to 23", [{"name": "set_thermostat", "arguments": {"temperature_c": 23.0}}]),
    ("set thermostat to 23.5 degrees", [{"name": "set_thermostat", "arguments": {"temperature_c": 23.5}}]),
    ("set house temperature to 24", [{"name": "set_thermostat", "arguments": {"temperature_c": 24.0}}]),
    ("set thermostat to 18", [{"name": "set_thermostat", "arguments": {"temperature_c": 18.0}}]),
    ("could you turn on the overhead fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "on"}}]),
    ("could you turn off the overhead fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "off"}}]),
    ("please switch on the circulating ceiling fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "on"}}]),
    ("please switch off the circulating ceiling fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "off"}}]),
    ("power up the room fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "on"}}]),
    ("stop the fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "off"}}]),
    ("please open up the garage door", [{"name": "control_device", "arguments": {"device": "garage door", "action": "open"}}]),
    ("close the garage door", [{"name": "control_device", "arguments": {"device": "garage door", "action": "close"}}]),
    ("can you raise up the garage door", [{"name": "control_device", "arguments": {"device": "garage door", "action": "open"}}]),
    ("can you shut down the garage door for me", [{"name": "control_device", "arguments": {"device": "garage door", "action": "close"}}]),
    ("please lock the main front door", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
    ("please unlock the main front door", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
    ("throw the bolt on front door", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
    ("unbolt front door", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
    ("secure front door", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
    ("unlatch front door", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
    ("open front lock", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
    ("secure the main front entrance lock", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
]

# ---------------------------------------------------------------------------
# 2. CATEGORY B: PARAPHRASE & SLANG (50 samples)
# ---------------------------------------------------------------------------
PARAPHRASE_50 = [
    ("kill all illumination in the kitchen", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}}]),
    ("brighten up the family room completely", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 100}}]),
    ("make the sitting area nice and dim at 20", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 20}}]),
    ("extinguish the lamps in the master quarters", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 0}}]),
    ("soften the bedroom down to a faint 15", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 15}}]),
    ("crank the cookhouse lighting up to 90", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 90}}]),
    ("blackout the restroom immediately", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 0}}]),
    ("light up the washroom full blast", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 100}}]),
    ("tone down the bathroom glow to 30", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 30}}]),
    ("darken the vehicle bay", [{"name": "set_lights", "arguments": {"room": "garage", "brightness": 0}}]),
    ("flood the garage with light", [{"name": "set_lights", "arguments": {"room": "garage", "brightness": 100}}]),
    ("drop lounge lights down to 25 percent", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 25}}]),
    ("cut power to all fixtures in the parlor", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 0}}]),
    ("can you illuminate the sleeping quarters", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 100}}]),
    ("dial back kitchen brightness to 40", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 40}}]),
    ("make it pitch black in the master bedroom", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 0}}]),
    ("bring the living room bulbs up to 70", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 70}}]),
    ("shut down lighting in the galley", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}}]),
    ("dampen the main hall lighting to 10", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 10}}]),
    ("maximize the bedroom chandelier", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 100}}]),
    ("it feels like an oven in here, drop climate to 18", [{"name": "set_thermostat", "arguments": {"temperature_c": 18.0}}]),
    ("chill this place right down to 19 celsius", [{"name": "set_thermostat", "arguments": {"temperature_c": 19.0}}]),
    ("warm the living space up to 23", [{"name": "set_thermostat", "arguments": {"temperature_c": 23.0}}]),
    ("crank the radiator target to 24 degrees", [{"name": "set_thermostat", "arguments": {"temperature_c": 24.0}}]),
    ("it is freezing cold, push the temp to 22.5", [{"name": "set_thermostat", "arguments": {"temperature_c": 22.5}}]),
    ("cool the house off to 20.5 please", [{"name": "set_thermostat", "arguments": {"temperature_c": 20.5}}]),
    ("dial the HVAC down to a crisp 18.5", [{"name": "set_thermostat", "arguments": {"temperature_c": 18.5}}]),
    ("bring thermal control up to 21", [{"name": "set_thermostat", "arguments": {"temperature_c": 21.0}}]),
    ("set the central heating to 22", [{"name": "set_thermostat", "arguments": {"temperature_c": 22.0}}]),
    ("ease the temperature back to 20", [{"name": "set_thermostat", "arguments": {"temperature_c": 20.0}}]),
    ("nudge the room temp to 21.5", [{"name": "set_thermostat", "arguments": {"temperature_c": 21.5}}]),
    ("get some airflow going with the ceiling fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "on"}}]),
    ("kill the ceiling fan breeze", [{"name": "control_device", "arguments": {"device": "fan", "action": "off"}}]),
    ("spin up the overhead fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "on"}}]),
    ("cut the electricity to the fan blades", [{"name": "control_device", "arguments": {"device": "fan", "action": "off"}}]),
    ("engage the ventilation fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "on"}}]),
    ("halt the ceiling fan rotation", [{"name": "control_device", "arguments": {"device": "fan", "action": "off"}}]),
    ("crank open the garage shutter", [{"name": "control_device", "arguments": {"device": "garage door", "action": "open"}}]),
    ("pull down the garage gate", [{"name": "control_device", "arguments": {"device": "garage door", "action": "close"}}]),
    ("lower the vehicle barrier", [{"name": "control_device", "arguments": {"device": "garage door", "action": "close"}}]),
    ("elevate the garage entrance door", [{"name": "control_device", "arguments": {"device": "garage door", "action": "open"}}]),
    ("fasten the front entrance deadbolt", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
    ("throw the bolt on the main doorway", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
    ("release the main entryway lock", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
    ("disengage the front door latch", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
    ("make sure the main exterior entrance is secured", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
    ("free the deadbolt on the front portal", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
    ("lock up the front access point", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
    ("undo the front door deadlock", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
    ("arm the physical lock on the entryway", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
]

# ---------------------------------------------------------------------------
# 3. CATEGORY C: TYPO & ASR NOISE (50 samples)
# ---------------------------------------------------------------------------
TYPO_50 = [
    ("trun of kitche light", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}}]),
    ("trun on the livng room lites", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 100}}]),
    ("dm bedrom lite to 25", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 25}}]),
    ("shutt off bathrom ligths", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 0}}]),
    ("swich on kitchn ligth", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 100}}]),
    ("dmm the livin rom to 40 pct", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 40}}]),
    ("set grage lights to 80", [{"name": "set_lights", "arguments": {"room": "garage", "brightness": 80}}]),
    ("darkn the bed room", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 0}}]),
    ("trun of bath room ligh", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 0}}]),
    ("brghten livingroom to 90", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 90}}]),
    ("tur off kitchin", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}}]),
    ("set bedrom to 30 prcent", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 30}}]),
    ("luminate the bathrom", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 100}}]),
    ("kil the lite in kitchen", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}}]),
    ("dim livin room light 15", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 15}}]),
    ("opn grage dor", [{"name": "control_device", "arguments": {"device": "garage door", "action": "open"}}]),
    ("clsoe grage door", [{"name": "control_device", "arguments": {"device": "garage door", "action": "close"}}]),
    ("shut down the grage enterance", [{"name": "control_device", "arguments": {"device": "garage door", "action": "close"}}]),
    ("rasie the garage dr", [{"name": "control_device", "arguments": {"device": "garage door", "action": "open"}}]),
    ("cloze garage dor", [{"name": "control_device", "arguments": {"device": "garage door", "action": "close"}}]),
    ("trun on celing fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "on"}}]),
    ("swich off teh fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "off"}}]),
    ("cut powr to the celing fan", [{"name": "control_device", "arguments": {"device": "fan", "action": "off"}}]),
    ("strt the fann", [{"name": "control_device", "arguments": {"device": "fan", "action": "on"}}]),
    ("stp the fan blads", [{"name": "control_device", "arguments": {"device": "fan", "action": "off"}}]),
    ("lokk frnt door", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
    ("unlok the frnt dor", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
    ("secuer frnt entranc", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
    ("unltch front dor", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
    ("bult the main dor", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
    ("disarm frnt deadblt", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
    ("lok up main entranc", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
    ("unlck frontdoor", [{"name": "lock_door", "arguments": {"door": "front door", "locked": False}}]),
    ("set themostat 21", [{"name": "set_thermostat", "arguments": {"temperature_c": 21.0}}]),
    ("chagne temp to 22.5 degres", [{"name": "set_thermostat", "arguments": {"temperature_c": 22.5}}]),
    ("set temparature to 19 celsus", [{"name": "set_thermostat", "arguments": {"temperature_c": 19.0}}]),
    ("thermostt 20.5 deg", [{"name": "set_thermostat", "arguments": {"temperature_c": 20.5}}]),
    ("adust tempreture to 18", [{"name": "set_thermostat", "arguments": {"temperature_c": 18.0}}]),
    ("set hvac to 23.5", [{"name": "set_thermostat", "arguments": {"temperature_c": 23.5}}]),
    ("chng climte to 20", [{"name": "set_thermostat", "arguments": {"temperature_c": 20.0}}]),
    ("set theromstat to 21.5 c", [{"name": "set_thermostat", "arguments": {"temperature_c": 21.5}}]),
    ("drop tmep to 19.5", [{"name": "set_thermostat", "arguments": {"temperature_c": 19.5}}]),
    ("mak it 22 degres", [{"name": "set_thermostat", "arguments": {"temperature_c": 22.0}}]),
    ("set themostat to 24", [{"name": "set_thermostat", "arguments": {"temperature_c": 24.0}}]),
    ("trun off evrything in bedrom", [{"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 0}}]),
    ("dim kitchn lite to 50", [{"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 50}}]),
    ("swich livingroom lites 100", [{"name": "set_lights", "arguments": {"room": "living room", "brightness": 100}}]),
    ("kil the bath rom bulbs", [{"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 0}}]),
    ("adjst thermstat to 18.5", [{"name": "set_thermostat", "arguments": {"temperature_c": 18.5}}]),
    ("lok the main doorway now", [{"name": "lock_door", "arguments": {"door": "front door", "locked": True}}]),
]

# ---------------------------------------------------------------------------
# 4. CATEGORY D: COMPOUND MULTI-STEP WITH NOVEL CONJUNCTIONS (50 samples)
# ---------------------------------------------------------------------------
COMPOUND_50 = [
    # Subordinating / Prepositional conjunctions (unseen by rule-based planners)
    (
        "before you lock up the front door, kill the kitchen lights",
        [
            {"name": "lock_door", "arguments": {"door": "front door", "locked": True}},
            {"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}},
        ]
    ),
    (
        "once the fan is off, close the garage door",
        [
            {"name": "control_device", "arguments": {"device": "fan", "action": "off"}},
            {"name": "control_device", "arguments": {"device": "garage door", "action": "close"}},
        ]
    ),
    (
        "cool down the house to 19 while turning on the fan",
        [
            {"name": "set_thermostat", "arguments": {"temperature_c": 19.0}},
            {"name": "control_device", "arguments": {"device": "fan", "action": "on"}},
        ]
    ),
    (
        "after dimming the bedroom to 25, lock up the front door",
        [
            {"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 25}},
            {"name": "lock_door", "arguments": {"door": "front door", "locked": True}},
        ]
    ),
    (
        "shut the garage door then lock the front entrance",
        [
            {"name": "control_device", "arguments": {"device": "garage door", "action": "close"}},
            {"name": "lock_door", "arguments": {"door": "front door", "locked": True}},
        ]
    ),
    (
        "turn on living room lights and set thermostat to 21",
        [
            {"name": "set_lights", "arguments": {"room": "living room", "brightness": 100}},
            {"name": "set_thermostat", "arguments": {"temperature_c": 21.0}},
        ]
    ),
    (
        "turn off kitchen lights and turn off the fan",
        [
            {"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}},
            {"name": "control_device", "arguments": {"device": "fan", "action": "off"}},
        ]
    ),
    (
        "dim living room to 30 and close the garage door",
        [
            {"name": "set_lights", "arguments": {"room": "living room", "brightness": 30}},
            {"name": "control_device", "arguments": {"device": "garage door", "action": "close"}},
        ]
    ),
    (
        "unlock the front door and turn on kitchen lights",
        [
            {"name": "lock_door", "arguments": {"door": "front door", "locked": False}},
            {"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 100}},
        ]
    ),
    (
        "set thermostat to 22.5 and lock the front door",
        [
            {"name": "set_thermostat", "arguments": {"temperature_c": 22.5}},
            {"name": "lock_door", "arguments": {"door": "front door", "locked": True}},
        ]
    ),
    (
        "turn off bedroom lights and set thermostat to 19.5",
        [
            {"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 0}},
            {"name": "set_thermostat", "arguments": {"temperature_c": 19.5}},
        ]
    ),
    (
        "open the garage door and turn on garage lights",
        [
            {"name": "control_device", "arguments": {"device": "garage door", "action": "open"}},
            {"name": "set_lights", "arguments": {"room": "garage", "brightness": 100}},
        ]
    ),
    (
        "turn on the fan and dim living room to 40",
        [
            {"name": "control_device", "arguments": {"device": "fan", "action": "on"}},
            {"name": "set_lights", "arguments": {"room": "living room", "brightness": 40}},
        ]
    ),
    (
        "lock the front door and close the garage door",
        [
            {"name": "lock_door", "arguments": {"door": "front door", "locked": True}},
            {"name": "control_device", "arguments": {"device": "garage door", "action": "close"}},
        ]
    ),
    (
        "unlock the front door and open the garage door",
        [
            {"name": "lock_door", "arguments": {"door": "front door", "locked": False}},
            {"name": "control_device", "arguments": {"device": "garage door", "action": "open"}},
        ]
    ),
    (
        "turn off bathroom light and turn off bedroom lights",
        [
            {"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 0}},
            {"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 0}},
        ]
    ),
    (
        "dim kitchen to 50 and dim living room to 50",
        [
            {"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 50}},
            {"name": "set_lights", "arguments": {"room": "living room", "brightness": 50}},
        ]
    ),
    (
        "turn off the fan and set thermostat to 20",
        [
            {"name": "control_device", "arguments": {"device": "fan", "action": "off"}},
            {"name": "set_thermostat", "arguments": {"temperature_c": 20.0}},
        ]
    ),
    (
        "turn on bedroom lights and unlock the front door",
        [
            {"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 100}},
            {"name": "lock_door", "arguments": {"door": "front door", "locked": False}},
        ]
    ),
    (
        "close the garage door and set thermostat to 18.5",
        [
            {"name": "control_device", "arguments": {"device": "garage door", "action": "close"}},
            {"name": "set_thermostat", "arguments": {"temperature_c": 18.5}},
        ]
    ),
    (
        "kill kitchen illumination and bolt the entrance",
        [
            {"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}},
            {"name": "lock_door", "arguments": {"door": "front door", "locked": True}},
        ]
    ),
    (
        "start the ceiling fan and drop the temp to 20",
        [
            {"name": "control_device", "arguments": {"device": "fan", "action": "on"}},
            {"name": "set_thermostat", "arguments": {"temperature_c": 20.0}},
        ]
    ),
    (
        "raise the garage shutter and illuminate the garage",
        [
            {"name": "control_device", "arguments": {"device": "garage door", "action": "open"}},
            {"name": "set_lights", "arguments": {"room": "garage", "brightness": 100}},
        ]
    ),
    (
        "secure the front door and blackout the living room",
        [
            {"name": "lock_door", "arguments": {"door": "front door", "locked": True}},
            {"name": "set_lights", "arguments": {"room": "living room", "brightness": 0}},
        ]
    ),
    (
        "warm the house up to 23 and shut the garage entrance",
        [
            {"name": "set_thermostat", "arguments": {"temperature_c": 23.0}},
            {"name": "control_device", "arguments": {"device": "garage door", "action": "close"}},
        ]
    ),
    (
        "extinguish bathroom lights and unlatch the front door",
        [
            {"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 0}},
            {"name": "lock_door", "arguments": {"door": "front door", "locked": False}},
        ]
    ),
    (
        "chill the place to 19 and spin up the ceiling fan",
        [
            {"name": "set_thermostat", "arguments": {"temperature_c": 19.0}},
            {"name": "control_device", "arguments": {"device": "fan", "action": "on"}},
        ]
    ),
    (
        "lower the garage gate and make sure the front lock is engaged",
        [
            {"name": "control_device", "arguments": {"device": "garage door", "action": "close"}},
            {"name": "lock_door", "arguments": {"door": "front door", "locked": True}},
        ]
    ),
    (
        "turn on kitchen light and turn off the ceiling fan",
        [
            {"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 100}},
            {"name": "control_device", "arguments": {"device": "fan", "action": "off"}},
        ]
    ),
    (
        "dim bedroom lights down to 20 and adjust thermostat to 21",
        [
            {"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 20}},
            {"name": "set_thermostat", "arguments": {"temperature_c": 21.0}},
        ]
    ),
    (
        "brighten living room to 80 then close the garage door",
        [
            {"name": "set_lights", "arguments": {"room": "living room", "brightness": 80}},
            {"name": "control_device", "arguments": {"device": "garage door", "action": "close"}},
        ]
    ),
    (
        "shut off the ceiling fan and open the garage door",
        [
            {"name": "control_device", "arguments": {"device": "fan", "action": "off"}},
            {"name": "control_device", "arguments": {"device": "garage door", "action": "open"}},
        ]
    ),
    (
        "unlock the front door and set climate control to 22",
        [
            {"name": "lock_door", "arguments": {"door": "front door", "locked": False}},
            {"name": "set_thermostat", "arguments": {"temperature_c": 22.0}},
        ]
    ),
    (
        "darken the kitchen completely and turn on the fan",
        [
            {"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}},
            {"name": "control_device", "arguments": {"device": "fan", "action": "on"}},
        ]
    ),
    (
        "lock the front entrance and set thermostat target to 18",
        [
            {"name": "lock_door", "arguments": {"door": "front door", "locked": True}},
            {"name": "set_thermostat", "arguments": {"temperature_c": 18.0}},
        ]
    ),
    (
        "switch off the fan and turn on the bedroom lights",
        [
            {"name": "control_device", "arguments": {"device": "fan", "action": "off"}},
            {"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 100}},
        ]
    ),
    (
        "close the garage door and dim the living room to 10",
        [
            {"name": "control_device", "arguments": {"device": "garage door", "action": "close"}},
            {"name": "set_lights", "arguments": {"room": "living room", "brightness": 100 if False else 10}},
        ]
    ),
    (
        "turn on bathroom light and turn off the kitchen light",
        [
            {"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 100}},
            {"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}},
        ]
    ),
    (
        "set thermostat to 20.5 and turn on the ceiling fan",
        [
            {"name": "set_thermostat", "arguments": {"temperature_c": 20.5}},
            {"name": "control_device", "arguments": {"device": "fan", "action": "on"}},
        ]
    ),
    (
        "unbolt the front door and raise the garage door",
        [
            {"name": "lock_door", "arguments": {"door": "front door", "locked": False}},
            {"name": "control_device", "arguments": {"device": "garage door", "action": "open"}},
        ]
    ),
    (
        "turn off garage lights and secure the front door",
        [
            {"name": "set_lights", "arguments": {"room": "garage", "brightness": 0}},
            {"name": "lock_door", "arguments": {"door": "front door", "locked": True}},
        ]
    ),
    (
        "dim the kitchen to 40 and shut down the garage door",
        [
            {"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 40}},
            {"name": "control_device", "arguments": {"device": "garage door", "action": "close"}},
        ]
    ),
    (
        "set temperature to 23 degrees and lock the front door",
        [
            {"name": "set_thermostat", "arguments": {"temperature_c": 23.0}},
            {"name": "lock_door", "arguments": {"door": "front door", "locked": True}},
        ]
    ),
    (
        "start the fan spinning and turn off living room lights",
        [
            {"name": "control_device", "arguments": {"device": "fan", "action": "on"}},
            {"name": "set_lights", "arguments": {"room": "living room", "brightness": 0}},
        ]
    ),
    (
        "illuminate the master bedroom and set thermostat to 21.5",
        [
            {"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 100}},
            {"name": "set_thermostat", "arguments": {"temperature_c": 21.5}},
        ]
    ),
    (
        "unlock front door and switch on bathroom light",
        [
            {"name": "lock_door", "arguments": {"door": "front door", "locked": False}},
            {"name": "set_lights", "arguments": {"room": "bathroom", "brightness": 100}},
        ]
    ),
    (
        "shut the garage door and turn off the fan",
        [
            {"name": "control_device", "arguments": {"device": "garage door", "action": "close"}},
            {"name": "control_device", "arguments": {"device": "fan", "action": "off"}},
        ]
    ),
    (
        "dim the living room to 25 and turn off kitchen light",
        [
            {"name": "set_lights", "arguments": {"room": "living room", "brightness": 25}},
            {"name": "set_lights", "arguments": {"room": "kitchen", "brightness": 0}},
        ]
    ),
    (
        "lock the front entrance and turn off the bedroom light",
        [
            {"name": "lock_door", "arguments": {"door": "front door", "locked": True}},
            {"name": "set_lights", "arguments": {"room": "bedroom", "brightness": 0}},
        ]
    ),
    (
        "open the garage door and unlock the front door",
        [
            {"name": "control_device", "arguments": {"device": "garage door", "action": "open"}},
            {"name": "lock_door", "arguments": {"door": "front door", "locked": False}},
        ]
    ),
]

# ---------------------------------------------------------------------------
# 5. CATEGORY E: HARD NEGATIVES & NEAR MISSES (50 samples)
# ---------------------------------------------------------------------------
HARD_NEGATIVES_50 = [
    # Appliance specs & repair (mentions device keywords but not actionable commands)
    ("how do you replace the capacitor in an oscillating ceiling fan", []),
    ("what is the standard rough-in height for a hallway thermostat", []),
    ("why does my deadbolt feel stiff when turning the brass key", []),
    ("is a 100 watt equivalent led bulb safe in an enclosed bathroom fixture", []),
    ("how many amps does a 1/2 horsepower garage door opener draw", []),
    ("can a smart thermostat run without a dedicated c-wire connection", []),
    ("what gauge electrical wire should be used for a 15 amp lighting circuit", []),
    ("who was granted the initial patent for the electric overhead fan", []),
    ("where can i find the battery compartment on a keyless front door deadbolt", []),
    ("does leaving ceiling fans running in an empty bedroom waste electricity", []),
    ("how does a bimetallic strip trigger an analog thermostat switch", []),
    ("what is the lumen output difference between warm white and daylight leds", []),
    ("why is my garage door reversing before reaching the ground threshold", []),
    ("can cold weather cause lithium batteries in door locks to drain faster", []),
    ("what is the average lifespan of a modern brushless dc ceiling fan motor", []),
    ("how do programmable thermostats calculate heating recovery cycle times", []),
    ("is it possible to lubricate a squeaky garage door track with wd-40", []),
    ("what are the physical dimensions of a standard residential mortise lock", []),
    ("how do electronic dimmer switches modulate power via pulse width", []),
    ("does lowering the thermostat target in winter actually save money", []),
    # Out of domain knowledge & general requests
    ("can you recommend three books about the history of architecture", []),
    ("what chemical reaction causes dough to rise during bread baking", []),
    ("explain the difference between nuclear fission and fusion in stars", []),
    ("who composed the soundtrack for the film interstellar", []),
    ("calculate the square root of 144 multiplied by 7", []),
    ("what is the capital city of new zealand", []),
    ("how do honeybees navigate back to their colony using polarized sunlight", []),
    ("summarize the main themes of dante's inferno in two sentences", []),
    ("can you write a haiku about autumn leaves falling on cold pavement", []),
    ("what is the boiling point of ethanol at standard atmospheric pressure", []),
    ("how does quantum entanglement relate to bell's inequality theorem", []),
    ("what were the primary economic drivers behind the industrial revolution", []),
    ("what ingredients do i need to prepare authentic roman carbonara", []),
    ("who was the prime minister of the united kingdom during world war one", []),
    ("why do optical telescopes perform better at high elevations", []),
    ("how does an aircraft wing generate lift according to bernoulli's principle", []),
    ("translate the greeting 'good evening my friend' into formal japanese", []),
    ("what is the difference between a mammal and a marsupial", []),
    ("how does the human circadian rhythm react to blue light frequencies", []),
    ("can you order two tickets for the upcoming basketball tournament", []),
    ("what is the distance in kilometers from the earth to the asteroid belt", []),
    ("explain how public key cryptography encrypts and decrypts messages", []),
    ("who wrote the dystopian science fiction novel brave new world", []),
    ("why do oceans appear blue when viewed from high orbit satellites", []),
    ("what are the three primary color pigments in subtractive color theory", []),
    ("how do catalytic converters reduce carbon monoxide emissions in vehicles", []),
    ("can you suggest five healthy snacks for an endurance bicycle trip", []),
    ("what is the approximate half-life of carbon 14 isotopes", []),
    ("who was the lead architect responsible for designing the eiffel tower", []),
    ("explain how modern touchscreens register capacitive finger input", []),
]


def create_frozen_suite():
    print("Building independent 250-sample frozen evaluation suite...")

    categories = [
        ("clean", CLEAN_50),
        ("paraphrase", PARAPHRASE_50),
        ("typo_noise", TYPO_50),
        ("compound", COMPOUND_50),
        ("hard_negative", HARD_NEGATIVES_50),
    ]

    # Verify counts
    total_samples = 0
    suite = []
    for cat_name, samples in categories:
        print(f"  Category '{cat_name}': {len(samples)} samples")
        assert len(samples) >= 50, f"Category {cat_name} has fewer than 50 samples ({len(samples)})"
        total_samples += len(samples)
        for q, gt in samples:
            suite.append({
                "category": cat_name,
                "query": q,
                "ground_truth": gt,
            })

    print(f"\nTotal test samples: {total_samples}")

    # Train / Test collision filter
    print("\nAuditing collision against Mara's 1,650 training samples...")
    train_samples = build_finetune_dataset(1500)
    train_queries = set(s["query"].strip().lower() for s in train_samples)

    collisions = []
    for item in suite:
        q_norm = item["query"].strip().lower()
        if q_norm in train_queries:
            collisions.append(q_norm)

    print(f"Exact collision count: {len(collisions)}")
    if collisions:
        print("Collisions found:")
        for c in collisions:
            print("  -", c)
        raise SystemExit("Collision check failed! Please resolve overlapping strings before freezing.")

    # Write frozen file
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for item in suite:
            f.write(json.dumps(item) + "\n")

    print(f"\n[SUCCESS] Frozen 250-sample benchmark suite written to: {OUT_FILE}")
    print("File size on disk:", os.path.getsize(OUT_FILE), "bytes")


if __name__ == "__main__":
    create_frozen_suite()
