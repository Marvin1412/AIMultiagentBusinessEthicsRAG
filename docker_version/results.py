import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def process_file(filepath):
    """
    Read a CSV or Excel file and return
     - counts: DataFrame mit Zeilenindex 'runde' und Spalten ['kooperieren','nicht kooperieren']
     - desc_df: DataFrame mit den kombinierten entscheidungen pro runde
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(filepath)
    elif ext in ('.xls', '.xlsx'):
        df = pd.read_excel(filepath)
    else:
        print(f"Skipping unsupported file type: {filepath}")
        return None, None

    # Spalten ermitteln, die mit 'entscheidung' beginnen
    ents_cols = [c for c in df.columns if c.startswith('entscheidung')]
    if not ents_cols:
        print(f"Keine Entscheidungs-Spalten in {filepath} gefunden.")
        return None, None

    # 1) Sammle je runde alle Entscheidungswerte aus den gefundenen Spalten
    decisions_per_round = (
        df
        .groupby('runde')[ents_cols]
        .apply(lambda g: g.values.flatten().tolist())
    )

    # 2) Kombinierte Entscheidung: 'kooperieren' wenn alle Werte 'kooperieren', sonst 'nicht kooperieren'
    combined = decisions_per_round.apply(
        lambda decs: 'kooperieren'
        if len(decs) >= 2 and all(d == 'kooperieren' for d in decs)
        else 'nicht kooperieren'
    )

    # 3) DataFrame mit kombinierten Entscheidungen
    desc_df = combined.rename('entscheidung').reset_index()

    # 4) Zähle Häufigkeiten pro runde
    counts = (
        desc_df
        .groupby(['runde', 'entscheidung'])
        .size()
        .unstack(fill_value=0)
    )

    return counts, desc_df


def process_folder(folder):
    """
    Process all CSV/Excel files in a folder.
    Returns:
      - file_counts: dict of per-file count DataFrames
      - file_descs: dict of per-file desc DataFrames
      - total_counts: aggregated counts DataFrame über alle Dateien
      - all_descs: concateniertes desc_df aller Dateien (für Histogramm)
    """
    file_counts = {}
    file_descs = {}
    all_counts = []
    all_descs = []

    patterns = [os.path.join(folder, '*.csv'), os.path.join(folder, '*.xls*')]
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            counts, desc_df = process_file(filepath)
            if counts is None:
                continue
            fname = os.path.basename(filepath)
            file_counts[fname] = counts
            file_descs[fname] = desc_df
            all_counts.append(counts)
            all_descs.append(desc_df)

    total_counts = pd.concat(all_counts).groupby(level=0).sum() if all_counts else pd.DataFrame()
    all_descs_df = pd.concat(all_descs, ignore_index=True) if all_descs else pd.DataFrame(columns=['runde','entscheidung'])

    return file_counts, file_descs, total_counts, all_descs_df


def plot_cooperation_trends(total_counts, folder_name, output_dir):
    """
    Zeichnet einen Linienplot: x=runde, y=Anzahl 'kooperieren'
    """
    plt.figure()
    koop = total_counts.get('kooperieren', pd.Series(dtype=int))
    koop.plot(marker='o')
    plt.title(f"Anzahl kooperieren pro runde ({folder_name})")
    plt.xlabel('runde')
    plt.ylabel('Anzahl kooperieren')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"trend_kooperieren_{folder_name}.png"))
    plt.close()


def plot_cooperation_histogram(all_descs_no_mod, all_descs_with_mod, output_dir):
    """
    Zeichnet ein Overlaid-Histogramm:
      - Anzahl der runden mit N-mal 'kooperieren' pro Ordner
    """
    counts_no = (
        all_descs_no_mod
        .groupby('runde')['entscheidung']
        .apply(lambda s: (s == 'kooperieren').sum())
    )
    counts_wm = (
        all_descs_with_mod
        .groupby('runde')['entscheidung']
        .apply(lambda s: (s == 'kooperieren').sum())
    )

    plt.figure()
    plt.hist(
        [counts_no.values, counts_wm.values],
        bins=range(0, max(counts_no.max(), counts_wm.max()) + 2),
        label=['no_mod', 'with_mod'],
        alpha=0.7,
        edgecolor='black'
    )
    plt.title("Histogramm: Verteilung der Anzahl kooperieren pro runde")
    plt.xlabel('Anzahl kooperieren pro runde')
    plt.ylabel('Anzahl runden')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histogramm_kooperieren_beide_ordner.png"))
    plt.close()


def main():
    no_mod_folder = './no_mod'
    with_mod_folder = './with_mod'
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    nm_counts, nm_descs, nm_total, nm_all_descs = process_folder(no_mod_folder)
    wm_counts, wm_descs, wm_total, wm_all_descs = process_folder(with_mod_folder)

    plot_cooperation_trends(nm_total, 'no_mod', output_dir)
    plot_cooperation_trends(wm_total, 'with_mod', output_dir)
    plot_cooperation_histogram(nm_all_descs, wm_all_descs, output_dir)

    # Optional: Speichern
    nm_total.to_csv(os.path.join(output_dir, 'no_mod_aggregated.csv'))
    wm_total.to_csv(os.path.join(output_dir, 'with_mod_aggregated.csv'))

    print('Analyse und Plots abgeschlossen. PNGs in', output_dir)


if __name__ == '__main__':
    main()
