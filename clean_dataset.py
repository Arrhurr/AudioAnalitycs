import os
import pandas as pd

# ---- Config ----
INPUT_PATH = os.getenv("DATASET_PATH", "Dataset_with_Genres.csv")
OUTPUT_PATH = os.getenv("OUTPUT_PATH", "dataset_clean_tango.csv")

# Colonnes audio les plus fréquentes (Spotify-like). On gardera celles qui existent vraiment.
AUDIO_FEATURES_CANDIDATES = [
    "danceability", "energy", "loudness", "tempo", "valence",
    "speechiness", "acousticness", "instrumentalness", "liveness"
]

# Colonnes "meta" utiles si elles existent
META_CANDIDATES = ["name", "artists", "duration", "consolidates_genre_lists"]

def main():
    # 1) Charger
    df = pd.read_csv(INPUT_PATH)
    print("Dataset chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print("Colonnes disponibles:", list(df.columns)[:30], "..." if len(df.columns) > 30 else "")

    # 2) Trouver la colonne genre (selon le dataset ça peut varier)
    genre_col = None
    for c in ["genre", "genres", "track_genre", "playlist_genre", "consolidates_genre_lists"]:
        if c in df.columns:
            genre_col = c
            break

    if genre_col is None:
        print(f"Aucune colonne de genre trouvée (genre/genres/track_genre/playlist_genre).")
        print("Le script va seulement nettoyer les features sans filtrer Tango.")
    else:
        # 3) Filtrer Tango (ajuste le mot-clé si ton dataset utilise d'autres termes)
        df = df[df[genre_col].astype(str).str.contains("tango", case=False, na=False)]
        print(f" Filtre Tango sur '{genre_col}': {df.shape[0]} lignes restantes")

    # 4) Garder uniquement les colonnes qui existent
    audio_features = [c for c in AUDIO_FEATURES_CANDIDATES if c in df.columns]
    meta_cols = [c for c in META_CANDIDATES if c in df.columns]

    if not audio_features:
        raise ValueError(
            "Aucune feature audio trouvée. Vérifier les noms de colonnes du CSV"
        )

    df = df[meta_cols + audio_features].copy()

    # 5) Convertir en numérique + drop NA
    for col in audio_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df = df.dropna(subset=audio_features)
    after = len(df)
    print(f" Drop NA sur features audio: {before - after} lignes supprimées, {after} restantes")

    # 6) Supprimer doublons si track_id existe (sinon, sur un combo simple)
    if "track_id" in df.columns:
        df = df.drop_duplicates(subset=["track_id"])
    else:
        key = [c for c in ["track_name", "artist_name", "year"] if c in df.columns]
        if key:
            df = df.drop_duplicates(subset=key)

    # 7) Export CSV clean
    df.to_csv(OUTPUT_PATH, index=False)
    print(f" Dataset nettoyé exporté: {OUTPUT_PATH}")
    print(" Aperçu:")
    print(df.head())

if __name__ == "__main__":
    main()
