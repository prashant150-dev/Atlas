"""Disjoint train / held-out English corpus for the P1.1 honest healing ceiling.

Two requirements the P1 (Day-4) run violated:
  * eval must NOT overlap training (else ppl < teacher = memorization, not skill),
  * eval must cover many tokens (P1 used 63 -> ~1.6%/token noise).

So we hand-write generic, original prose on varied topics, split into a TRAIN
pool and a *separate* HELDOUT pool that shares no sentences. The healed student
trains on TRAIN windows and is judged only on HELDOUT — top-1 agreement on
unseen text is the real ceiling signal.
"""

from __future__ import annotations

# ~50 generic sentences across science, history, nature, daily life, ideas.
TRAIN_TEXT: tuple[str, ...] = (
    "Observation is the patient act of watching the world until it reveals a pattern.",
    "A theory earns trust only after it survives many honest attempts to break it.",
    "Water flows downhill because every drop follows the simplest available path.",
    "The seasons turn because the planet leans as it travels around its star.",
    "Bridges stand for centuries when their builders respect the weight of stone.",
    "A market is a crowd of strangers quietly agreeing on what things are worth.",
    "Maps shrink a country onto paper so a traveller can hold the road in one hand.",
    "Bread rises because tiny living cells breathe inside the warm and resting dough.",
    "The oldest stories were spoken aloud long before anyone learned to write them down.",
    "A river carves a canyon not by force but by refusing, for ages, to stop.",
    "Light bends as it enters water, which is why a straight stick looks broken.",
    "Farmers read the sky the way sailors once read the slow drift of the stars.",
    "A good question opens a door that a hundred confident answers had kept shut.",
    "Iron rusts when air and water are given enough time to work upon its surface.",
    "The printing press turned a rare and guarded book into a common household thing.",
    "Bees keep a hive warm in winter by trembling together in a tight living ball.",
    "A coastline looks longer the closer you measure it, down to every grain of sand.",
    "Trade routes carried spices and ideas with equal ease across the open desert.",
    "The moon raises the tide twice a day by pulling gently on the patient sea.",
    "A craftsman learns the grain of the wood before the chisel ever touches it.",
    "Clouds are simply rivers that have chosen, for a while, to walk through the air.",
    "Numbers let a shepherd keep a flock in his head without ever seeing the sheep.",
    "An echo is a sound that returns after touching a wall it could not pass through.",
    "Cities grow where two roads meet and travellers find a reason to stay the night.",
    "The compass needle remembers a direction the eyes alone could never have found.",
    "A seed holds, folded inside it, the whole shape of the tree it intends to become.",
    "Glass is made by heating ordinary sand until it forgets that it was ever solid.",
    "Historians argue because the past leaves more questions than it leaves answers.",
    "Salt was once so prized that armies were sometimes paid in nothing else at all.",
    "A telescope gathers faint old light that left its star before the city was built.",
    "Rain forests breathe out the air that distant grasslands quietly breathe back in.",
    "The wheel was a small idea that quietly rebuilt every road that came after it.",
    "A melody is remembered long after the words that once rode upon it are lost.",
    "Mountains rise slowly as two vast plates of rock lean their shoulders together.",
    "A library is a long conversation among people who will never meet in person.",
    "Frost draws delicate ferns on a window using only cold and a little moisture.",
    "Sailors trusted the steady star that alone refused to wander across the night.",
    "A clock divides the endless day into pieces small enough for people to share.",
    "Volcanoes build new land from the very heat that the old land tried to hide.",
    "The alphabet shrank a thousand pictures into a handful of reusable small marks.",
    "A spider rebuilds its torn web each morning without complaint or any blueprint.",
    "Coins carried the face of a king into pockets he would never live to visit.",
    "Snow muffles a city because each flake is mostly the silence of trapped air.",
    "A good teacher hands over not the answer but the courage to keep on looking.",
    "Deserts are cold at night because the dry air keeps none of the day's warmth.",
    "The first maps of the sky were drawn by people guessing at enormous distances.",
    "A canal lets a boat climb a hill one calm and patient step of water at a time.",
    "Memory keeps the useful and quietly lets the ordinary day slip out of reach.",
    "Wind is only the air hurrying from a crowded place toward an emptier one.",
    "Every harvest is a wager placed months earlier against an unknown sky.",
)

# ~18 DIFFERENT sentences, no overlap with TRAIN -> the honest test set.
HELDOUT_TEXT: tuple[str, ...] = (
    "A lighthouse spends every night warning ships about a danger it cannot move.",
    "Honey never truly spoils because almost nothing living can survive inside it.",
    "The tallest trees lift water upward against gravity through threads finer than hair.",
    "Ancient roads were built so straight that armies could march without ever pausing.",
    "A whisper carries across a curved stone gallery as if the wall were listening.",
    "Migrating birds inherit a map they were never taught and have never once seen.",
    "Paper folds remember every crease long after the original shape is gone.",
    "A drought teaches a valley exactly how much it had always taken for granted.",
    "The deepest caves hold weather of their own, with slow winds and steady rain.",
    "A violin shapes the same air that a shout would waste into something worth keeping.",
    "Glaciers record old winters in layers the way a tree records its quiet years.",
    "Markets crash when a shared and cheerful story suddenly stops being believed.",
    "A kite stays aloft by leaning against the very wind that is trying to push it down.",
    "Old harbours silt up slowly until the sea that built them can no longer enter.",
    "The brightest comets return on schedules longer than any single human life.",
    "A rumor travels faster than the truth because it asks so much less of the listener.",
    "Stone stairs wear into gentle curves under centuries of ordinary patient feet.",
    "A good harbor forgives a clumsy sailor that the open ocean never would.",
)
