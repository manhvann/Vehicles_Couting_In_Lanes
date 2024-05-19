coordinates = [
    "X: 715, Y: 540",
    "X: 1019, Y: 493",
    "X: 763, Y: 166",
    "X: 616, Y: 180",
    "X: 719, Y: 559"
]

converted_coordinates = []

for coord in coordinates:
    parts = coord.split(",")
    x_part = parts[0].split(":")[1].strip()
    y_part = parts[1].split(":")[1].strip()

    try:
        x = int(x_part)
        y = int(y_part)
        converted_coordinates.append([x, y])
    except ValueError:
        print(f"Invalid format for coordinate: {coord}")

print(converted_coordinates)
