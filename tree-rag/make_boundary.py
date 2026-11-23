"""Extract the Thatta boundary from a downloaded geoBoundaries / GADM zip.

Usage:  py make_boundary.py [path-to-zip]
Looks for a .zip in the data folder when no path is given.
Writes data/thatta_boundary.geojson.
"""
import glob
import io
import json
import os
import sys
import zipfile

DATA_DIR = "data"
OUT_FILE = os.path.join(DATA_DIR, "thatta_boundary.geojson")
NAMES = ("thatta", "tatta")
TOWN = (24.7476, 67.9236)  # lat, lon of Thatta town


def find_zip():
    if len(sys.argv) > 1:
        return sys.argv[1]
    zips = glob.glob(os.path.join(DATA_DIR, "*.zip"))
    if len(zips) != 1:
        sys.exit(f"Found {len(zips)} zip files in {DATA_DIR}/. Pass the path: py make_boundary.py data\\file.zip")
    return zips[0]


def features_from_zip(archive):
    """Return (source_name, list of GeoJSON features) for every readable file in the zip."""
    found = []
    for name in archive.namelist():
        lower = name.lower()
        if lower.endswith((".geojson", ".json")):
            try:
                data = json.loads(archive.read(name).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                continue
            if data.get("type") == "FeatureCollection":
                found.append((name, data.get("features", [])))
        elif lower.endswith(".shp"):
            try:
                import shapefile  # pyshp
            except ImportError:
                print(f"  {name}: shapefile found, but pyshp is not installed (pip install pyshp). Skipping.")
                continue
            base = name[:-4]
            try:
                reader = shapefile.Reader(
                    shp=io.BytesIO(archive.read(base + ".shp")),
                    shx=io.BytesIO(archive.read(base + ".shx")),
                    dbf=io.BytesIO(archive.read(base + ".dbf")),
                )
            except KeyError:
                continue
            feats = [
                {"type": "Feature", "properties": dict(rec.record.as_dict()), "geometry": rec.shape.__geo_interface__}
                for rec in reader.shapeRecords()
            ]
            found.append((name, feats))
    return found


def matches(feature):
    values = [str(v).lower() for v in (feature.get("properties") or {}).values() if isinstance(v, str)]
    return any(n in value for value in values for n in NAMES)


def polygons(geometry):
    if not geometry:
        return []
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return list(geometry["coordinates"])
    return []


def in_ring(lon, lat, ring):
    inside, j = False, len(ring) - 1
    for i in range(len(ring)):
        xi, yi, xj, yj = ring[i][0], ring[i][1], ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def point_inside(features, lat, lon):
    for feature in features:
        for rings in polygons(feature.get("geometry")):
            if in_ring(lon, lat, rings[0]) and not any(in_ring(lon, lat, h) for h in rings[1:]):
                return True
    return False


def main():
    path = find_zip()
    print("Zip:", path)
    with zipfile.ZipFile(path) as archive:
        print("Files inside:")
        for name in archive.namelist():
            print("  ", name)
        sources = features_from_zip(archive)

    if not sources:
        sys.exit("No readable GeoJSON or shapefile found in the zip.")

    chosen = []
    for name, feats in sources:
        hits = [f for f in feats if matches(f)]
        print(f"\n{name}: {len(feats)} features, {len(hits)} match 'Thatta'")
        for f in hits:
            props = {k: v for k, v in (f.get("properties") or {}).items() if isinstance(v, (str, int, float))}
            print("   ", props)
        if hits and not chosen:
            chosen = hits  # first file with a match; later files are usually the same data

    if not chosen:
        sys.exit("\nNo feature named Thatta/Tatta. Send me the 'Files inside' list above.")

    collection = {"type": "FeatureCollection", "features": chosen}
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as file:
        json.dump(collection, file)

    polys = sum(len(polygons(f.get("geometry"))) for f in chosen)
    size_kb = os.path.getsize(OUT_FILE) / 1024
    print(f"\nWrote {OUT_FILE}: {len(chosen)} feature(s), {polys} polygon(s), {size_kb:.0f} KB")
    print("Thatta town inside boundary:", point_inside(chosen, *TOWN))


if __name__ == "__main__":
    main()