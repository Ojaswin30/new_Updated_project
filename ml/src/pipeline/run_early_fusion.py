import argparse
from ml.src.pipeline.early_fusion_pipeline import EarlyFusionPipeline


def main():

    parser = argparse.ArgumentParser(
        description="Symbolic Early Fusion Runner"
    )

    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)

    args = parser.parse_args()

    pipeline = EarlyFusionPipeline()

    output = pipeline.run(args.image, args.text)

    print("\n--- EARLY FUSION OUTPUT ---\n")

    print("Image Signal:")
    print(output["image_signal"])

    print("\nText Signal:")
    print(output["text_signal"])

    print("\nFinal Constraints:")
    print(output["final_constraints"])

    print("\nGenerated SQL:")
    print(output["query"]["sql"])
    print("Params:", output["query"]["params"])

    print("\nFusion Statistics:")
    print(output["statistics"])


if __name__ == "__main__":
    print("CLIP inference executing...")
    main()