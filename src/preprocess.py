import pandas as pd
import os

MIN_WEEKS_THRESHOLD = 20


def discover_states(df, state_col='state'):
    """
    Returns a list of valid state names found in the dataframe.
    Filters out states with insufficient data.
    """
    states = df[state_col].dropna().unique().tolist()
    print(f"\n[Preprocess] Found {len(states)} unique states: {states}")
    return states


def check_data_sufficiency(state_df, state_name,
                            min_weeks=MIN_WEEKS_THRESHOLD):
    """
    Returns True if the state has enough data to model.
    Prints a warning and returns False if not.
    """
    n = len(state_df)
    if n < min_weeks:
        print(f"  [SKIP] {state_name}: only {n} weeks of data "
              f"(minimum required: {min_weeks}). Skipping.")
        return False
    return True


def process_state(df, state_name, state_col='state',
                  date_col='date', sales_col='sales'):
    """
    Processes a single state's data:
    - Filters rows for this state
    - Parses and sorts dates
    - Resamples to weekly (W = Sunday-ending)
    - Fills gaps via linear interpolation + forward/back fill
    - Returns a clean DataFrame with columns [date, sales]
    """
    state_df = df[df[state_col] == state_name].copy()
    state_df[date_col] = pd.to_datetime(state_df[date_col])
    state_df = state_df.sort_values(date_col)
    state_df = state_df.set_index(date_col)[[sales_col]]
    state_df = state_df.resample('W').sum()

    # Fill the full date range with zeros then interpolate
    full_range = pd.date_range(
        start=state_df.index.min(),
        end=state_df.index.max(), freq='W')
    state_df = state_df.reindex(full_range)
    state_df[sales_col] = (
        state_df[sales_col]
        .interpolate(method='linear')
        .ffill()
        .bfill()
    )
    state_df = state_df.reset_index()
    state_df.columns = ['date', 'sales']
    return state_df


def load_and_process_all(filepath='data/sales_data.xlsx',
                          output_dir='data/processed/'):
    """
    Master function: loads the Excel file, discovers all states,
    processes each one, checks data sufficiency, and saves CSVs.

    Returns:
        dict: { state_name: DataFrame } for all valid states
    """
    if not os.path.exists(filepath):
        print(f"[Preprocess] Error: {filepath} not found.")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    print(f"[Preprocess] Loading: {filepath}")
    raw = pd.read_excel(filepath)

    # Normalize column names to lowercase
    raw.columns = [c.strip().lower() for c in raw.columns]

    # Auto-detect column names (flexible)
    col_map = {}
    for col in raw.columns:
        if 'date' in col or 'week' in col:
            col_map['date'] = col
        elif 'state' in col or 'region' in col or 'location' in col:
            col_map['state'] = col
        elif 'sale' in col or 'revenue' in col or 'amount' in col:
            col_map['sales'] = col

    missing = [k for k in ['date', 'state', 'sales']
               if k not in col_map]
    if missing:
        raise ValueError(
            f"Could not detect columns: {missing}. "
            f"Found columns: {list(raw.columns)}")

    print(f"[Preprocess] Column mapping: {col_map}")

    states = discover_states(raw, state_col=col_map['state'])

    state_dict = {}
    skipped    = []
    saved      = []

    for state in states:
        processed = process_state(
            raw, state,
            state_col=col_map['state'],
            date_col=col_map['date'],
            sales_col=col_map['sales']
        )
        if not check_data_sufficiency(processed, state):
            skipped.append(state)
            continue

        save_path = os.path.join(output_dir, f"{state}.csv")
        processed.to_csv(save_path, index=False)
        saved.append(state)
        state_dict[state] = processed
        print(f"  [Saved] {save_path} - {len(processed)} weeks")

    # -- Summary --
    print(f"\n{'='*50}")
    print(f"  Preprocessing Summary")
    print(f"{'='*50}")
    print(f"  Total states found : {len(states)}")
    print(f"  States processed   : {len(saved)}  -> {saved}")
    print(f"  States skipped     : {len(skipped)} -> {skipped}")
    print(f"{'='*50}\n")

    return state_dict


if __name__ == '__main__':
    print("Loading and processing data...")
    state_dict = load_and_process_all()
    print("Preprocessing complete.")
