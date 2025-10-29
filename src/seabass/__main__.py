import json
import os

import click as click
import runez

from seabass.core.models.mms import estimated_mean_mushra_score_file


@runez.click.command()
@runez.click.version()
@click.option("-t", "--test-path", required=True, help="Path to test signal file")
@click.option("-r", "--ref-path", required=True, help="Path to reference signal file")
@click.option("-o", "--output-path", default=None, help="Path to output file")
@click.option(
    "-f", "--resampy-filter", default="kaiser_best", help="Resampy filter type"
)
def main(
    test_path: str,
    ref_path: str,
    output_path: str = None,
    resampy_filter: str = "kaiser_best",
):
    mms = estimated_mean_mushra_score_file(
        test_path=test_path, ref_path=ref_path, resampy_filter=resampy_filter
    )

    print(f"Estimated Mean MUSHRA Score: {mms:3.2f}")

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(
                {
                    "estimated_mean_mushra_score": mms,
                    "test_path": test_path,
                    "ref_path": ref_path,
                    "resampy_filter": resampy_filter,
                },
                f,
                indent=4,
            )


if __name__ == "__main__":
    main()
