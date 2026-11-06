from pathlib import Path

pair_path = Path(
    r"C:\Users\osun2\IdeaProjects\multimodal-garment-designer\assets\data\vitonhd\test_pairs.txt"
)

print(pair_path.read_text())
print("Total lines:", len(pair_path.read_text().splitlines()))