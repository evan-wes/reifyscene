"""The labels the semantics stage chooses among.

- **Things** (``THING_VOCAB``): COCO-80's object classes, and the 346 categories of LVIS v1's
  *frequent* tier that COCO lacks (``LVIS_FREQUENT``). COCO alone has no fish: on the DAVIS
  gold-fish clip the four fish were named "bird", and once background classes were offered,
  "water". All of LVIS (1,203 categories) names the fish, but its rare tail outbids everything:
  people became "CD player" and backdrop "plume" on the ten DAVIS slice runs (2026-10-17). The
  frequent tier names the fish "fish" and the people "person".
- **Background** (``BACKGROUND_VOCAB``): what a scene is made of rather than what is in it. SAM 2's
  automatic masks propose floors, lawns, sky and crowds as readily as things; offered these, the
  semantics stage can say so, and the graph gives such a track to the background, as ConceptGraphs
  drops its background classes (``skip_bg``). COCO's classes beat a background class on equal
  terms; LVIS's must win by a margin (``siglip_semantics.DEFAULT_BACKGROUND_MARGIN``), because
  its tail scores close to any backdrop.
- **People** (``PERSON_VOCAB``): a track the person detector already calls a person is named
  among COCO's classes and the background ones only -- with LVIS's clothing on offer, people
  were named "dress" and "jean".

``LVIS_FREQUENT`` is the frequent tier of the LVIS v1 category names (Gupta, Dollar, Girshick,
"LVIS: A Dataset for Large Vocabulary Instance Segmentation", CVPR 2019;
https://www.lvisdataset.org, annotations CC BY 4.0), as listed in detectron2's
``lvis_v1_categories.py`` (Apache-2.0), with underscores as spaces, less the names COCO-80 or
``BACKGROUND_VOCAB`` already hold.
"""

from __future__ import annotations

from reifyscene.perception.objects.detector import COCO_INSTANCE_CATEGORY_NAMES

COCO_THINGS = [c for c in COCO_INSTANCE_CATEGORY_NAMES if c not in ('__background__', 'N/A')]

BACKGROUND_VOCAB = [
    'floor', 'wall', 'ceiling', 'ground', 'grass', 'dirt', 'sand', 'snow', 'gravel', 'road',
    'pavement', 'sidewalk', 'sky', 'water', 'sea', 'wave', 'river', 'building', 'fence', 'tree',
    'bush', 'mountain', 'hill', 'rock', 'field', 'carpet', 'crowd',
]

LVIS_FREQUENT = [
    'air conditioner', 'alarm clock', 'antenna', 'apron', 'armchair', 'trash can', 'avocado',
    'awning', 'baby buggy', 'ball', 'balloon', 'bandanna', 'banner', 'barrel', 'baseball base',
    'baseball', 'baseball cap', 'basket', 'bath mat', 'bath towel', 'bathtub', 'beanie',
    'bedspread', 'beef (food)', 'beer bottle', 'bell', 'bell pepper', 'belt', 'belt buckle',
    'visor', 'billboard', 'birthday cake', 'blackboard', 'blanket', 'blender', 'blinker', 'blouse',
    'blueberry', 'bolt', 'boot', 'bow (decorative ribbons)', 'bow-tie', 'box', 'bracelet', 'bread',
    'bridal gown', 'bucket', 'bun', 'buoy', 'bus (vehicle)', 'butter', 'button', 'cab (taxi)',
    'cabinet', 'calendar', 'camera', 'can', 'candle', 'candle holder', 'cap (headwear)',
    'bottle cap', 'car (automobile)', 'railcar (part of a train)', 'tote bag', 'cauliflower',
    'celery', 'cellular telephone', 'chandelier', 'choker', 'chopping board', 'chopstick',
    'Christmas tree', 'cigarette', 'cistern', 'clock tower', 'coaster', 'coat', 'coffee maker',
    'coffee table', 'computer keyboard', 'condiment', 'cone', 'control', 'cookie',
    'cooler (for food)', 'cork (bottle plug)', 'edible corn', 'cowboy hat', 'crate', 'crossbar',
    'crumb', 'cucumber', 'cupboard', 'cupcake', 'curtain', 'cushion', 'deck chair', 'desk', 'dish',
    'dishtowel', 'dishwasher', 'dispenser', 'Dixie cup', 'dog collar', 'doll', 'doorknob',
    'doughnut', 'drawer', 'dress', 'dress suit', 'dresser', 'duck', 'duffel bag', 'earphone',
    'earring', 'egg', 'fan', 'faucet', 'figurine', 'fire alarm', 'fire engine',
    'fire extinguisher', 'fireplace', 'fireplug', 'fish', 'flag', 'flagpole', 'flip-flop (sandal)',
    'flower arrangement', 'frying pan', 'glass (drink container)', 'glove', 'goggles', 'grape',
    'green bean', 'green onion', 'grill', 'guitar', 'hairbrush', 'ham', 'hair dryer', 'hand towel',
    'handle', 'hat', 'headband', 'headboard', 'headlight', 'helmet', 'hinge',
    'home plate (baseball)', 'fume hood', 'hook', 'hose', 'polar bear', 'iPod', 'jacket', 'jar',
    'jean', 'jersey', 'key', 'kitchen sink', 'knee pad', 'knob', 'ladder', 'lamb (animal)', 'lamp',
    'lamppost', 'lampshade', 'lanyard', 'laptop computer', 'latch', 'lemon', 'lettuce',
    'license plate', 'life buoy', 'life jacket', 'lightbulb', 'lime', 'log',
    'speaker (stero equipment)', 'magazine', 'magnet', 'mailbox (at home)', 'manhole', 'map',
    'marker', 'mask', 'mast', 'mattress', 'microphone', 'microwave oven', 'milk', 'minivan',
    'mirror', 'monitor (computer equipment) computer monitor', 'motor', 'motor scooter',
    'mound (baseball)', 'mouse (computer equipment)', 'mousepad', 'mug', 'mushroom', 'napkin',
    'necklace', 'necktie', 'newspaper', 'notebook', 'nut', 'oar', 'onion', 'orange (fruit)',
    'ottoman', 'paddle', 'painting', 'pajamas', 'pan (for cooking)', 'paper plate', 'paper towel',
    'pastry', 'pear', 'pen', 'pencil', 'pepper', 'piano', 'pickle', 'pickup truck', 'pillow',
    'pineapple', 'pipe', 'pitcher (vessel for liquid)', 'place mat', 'plate', 'pole', 'polo shirt',
    'pop (soda)', 'poster', 'pot', 'flowerpot', 'potato', 'printer', 'propeller', 'quilt',
    'radiator', 'rearview mirror', 'reflector', 'remote control', 'ring', 'rubber band',
    'plastic bag', 'saddle (on an animal)', 'saddle blanket', 'sail', 'salad', 'saltshaker',
    'sandal (type of shoe)', 'saucer', 'sausage', 'scale (measuring instrument)', 'scarf',
    'scoreboard', 'scrubbing brush', 'shirt', 'shoe', 'shopping bag', 'short pants',
    'shoulder bag', 'shower head', 'shower curtain', 'signboard', 'ski', 'ski boot', 'ski parka',
    'ski pole', 'skirt', 'soap', 'soccer ball', 'sock', 'sofa', 'soup', 'spatula', 'spectacles',
    'statue (sculpture)', 'steering wheel', 'stirrup', 'stool', 'brake light', 'stove', 'strap',
    'straw (for drinking)', 'strawberry', 'street sign', 'streetlight', 'suit (clothing)',
    'sunglasses', 'sweater', 'sweatshirt', 'swimsuit', 'table', 'tablecloth', 'tag', 'taillight',
    'tank (storage vessel)', 'tank top (clothing)', 'tape (sticky cloth or paper)', 'tarp',
    'teapot', 'telephone', 'telephone pole', 'television set', 'tennis ball', 'thermostat',
    'tinfoil', 'tissue paper', 'toaster oven', 'toilet tissue', 'tomato', 'tongs', 'toothpaste',
    'toothpick', 'cover', 'towel', 'towel rack', 'toy', 'trailer truck',
    'train (railroad vehicle)', 'tray', 'tripod', 'trousers', 'underwear', 'urinal', 'vent',
    'vest', 'wall socket', 'wallet', 'watch', 'water bottle', 'watermelon', 'weathervane',
    'wet suit', 'wheel', 'windshield wiper', 'wine bottle', 'wineglass', 'blinder (for horses)',
    'wristband', 'wristlet',
]

THING_VOCAB = COCO_THINGS + LVIS_FREQUENT
DEFAULT_VOCAB = THING_VOCAB + BACKGROUND_VOCAB
PERSON_VOCAB = COCO_THINGS + BACKGROUND_VOCAB
