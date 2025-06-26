"""
Based on:
EXPLORING THE INNER MECHANISMS OF LARGE GENERATIVE
MUSIC MODELS (2024) Vásquez et al.
"""

import os


MOODS = ["happy", "sad", "energetic", "calm", "dramatic", "uplifting", "tense"]
GENRES = ["jazz", "classical", "rock", "pop", "edm", "hip hop", "country"]
COMPOSITIONS = ["song", "jingle", "piece"]
INSTRUMENTS = ["piano", "guitar", "trumpet", "violin"]
TEMPOS = ["fast", "slow", "moderate"]
VERBS = ["Compose", "Create", "Generate"]
TOTAL_PROMPTS = len(MOODS) * len(GENRES) * len(TEMPOS) * len(COMPOSITIONS) * len(VERBS)


def generate_prompt(verb, mood, genre, composition, instrument, tempo) -> str:
    """
    Generate a music prompt based on the given parameters.
    """

    return f"{verb} a {mood} {genre} {composition} with a {instrument} melody. Use a {tempo} tempo."


def generate_instrument_prompt(instrument) -> None:
    """
    Helper function to generate prompts for a specific instrument. Saves to file.
    """

    prompts = []

    for verb in VERBS:
        for mood in MOODS:
            for genre in GENRES:
                for composition in COMPOSITIONS:
                    for tempo in TEMPOS:
                        prompt = generate_prompt(verb, mood, genre, composition, instrument, tempo)
                        prompts.append(prompt)

    assert len(prompts) == TOTAL_PROMPTS, f"Expected {TOTAL_PROMPTS} prompts, but got {len(prompts)}"

    # Write prompts to a file
    with open(f"{args.output_dir}/prompts_{instrument}.txt", 'w') as f:
        for prompt in prompts:
            f.write(prompt + '\n')


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating {TOTAL_PROMPTS} prompts per instrument ({len(INSTRUMENTS)} instruments)")

    for instrument in INSTRUMENTS:
        generate_instrument_prompt(instrument)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate music prompts based on mood, genre, instrument, and tempo.")
    # specify output file
    parser.add_argument('--output-dir', default="prompts", type=str, help='Output directory')
    args = parser.parse_args()

    main(args)
