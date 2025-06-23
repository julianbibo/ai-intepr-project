"""
Based on:
EXPLORING THE INNER MECHANISMS OF LARGE GENERATIVE
MUSIC MODELS (2024) Vásquez et al.
"""

import os


MOODS = ["happy", "sad", "energetic", "calm", "dramatic"]
GENRES = ["jazz", "classical", "rock", "pop", "edm", "hip hop"]
COMPOSITIONS = ["song", "jingle", "piece"]
INSTRUMENTS = ["piano", "guitar", "trumpet", "violin"]
TEMPOS = ["fast", "slow", "moderate"]


def generate_prompt(mood, genre, composition, instrument, tempo):
    """
    Generate a music prompt based on the given parameters.
    """

    return f"Compose a {mood} {genre} {composition} with a {instrument} melody. Use a {tempo} tempo."


def generate_instrument_prompt(instrument):
    """
    Helper function to generate prompts for a specific instrument. Saves to file.
    """

    total_size = len(MOODS) * len(GENRES) * len(TEMPOS) * len(COMPOSITIONS)
    prompts = []

    for mood in MOODS:
        for genre in GENRES:
            for composition in COMPOSITIONS:
                for tempo in TEMPOS:
                    prompt = generate_prompt(mood, genre, composition, instrument, tempo)
                    prompts.append(prompt)

    assert len(prompts) == total_size, f"Expected {total_size} prompts, but got {len(prompts)}"

    # Write prompts to a file
    with open(f"{args.output_dir}/prompts_{instrument}.txt", 'w') as f:
        for prompt in prompts:
            f.write(prompt + '\n')


def main(args):
    # Example parameters

    os.makedirs(args.output_dir, exist_ok=True)

    total_size = len(MOODS) * len(GENRES) * len(INSTRUMENTS) * len(TEMPOS)
    print(f"Generating {total_size} prompts per instrument ({len(INSTRUMENTS)} instruments)")

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
