import numpy as np
import pandas as pd
import os
import warnings
import traceback
import matplotlib.pyplot as plt

# --- Import necessary functions from HelperFuncs ---
try:
    from HelperFuncs import load_file, save_file, random_partition, safe_mean, sort_df_by_trip_time, find_seg_start_idx
except ImportError:
    print("Error: Could not import from HelperFuncs. Ensure HelperFuncs.py is accessible.")
    def load_file(filepath, **kwargs): raise NotImplementedError("HelperFuncs not found")
    def save_file(data, path, file_name, format='pickle', index=False): raise NotImplementedError("HelperFuncs not found")
    def random_partition(total, num_parts, random_state=42): raise NotImplementedError("HelperFuncs not found")
    def safe_mean(val): raise NotImplementedError("HelperFuncs not found")
    def sort_df_by_trip_time(df, trip_id_col='trip_id', time_col='timestamp'): raise NotImplementedError("HelperFuncs not found")

# --- Segmentation Functions ---

def find_natural_stops(df, trip_id_col="trip_id", speed_array_col="speed_array",
                       odo_col="current_odo", stop_indicator=-1,
                       max_speed_in_stop_kph=2.0, stationary_speed_thr_kph=0.5,
                       stationary_sample_pct_thr=0.90, thr_dist_km=0.05, debug=False):
    """
    Detects stationary periods using a fully vectorized, index-safe method.
    This function is significantly faster than loop-based approaches.

    It adds a 'stop_flag' column to the DataFrame and prints summary statistics.
    When debug=True, it prints details about rows where stop conditions mismatch.
    """
    print("\n--- Finding Natural Stops (Vectorized, High-Performance Logic) ---")
    if debug:
        print(" - Debug mode is ON. Mismatches in stop conditions will be reported.")
    
    processed_trips = []
    
    for trip_id, trip_df in df.groupby(trip_id_col, sort=False):
        g = trip_df.reset_index(drop=True)

        max_speed_vals = np.array([
            np.nanmax(a) if isinstance(a, (list, np.ndarray)) and len(a) > 0 else np.inf
            for a in g[speed_array_col]
        ])
        
        pct_stationary_vals = np.array([
            (np.sum(np.array(a) < stationary_speed_thr_kph) / len(a))
            if isinstance(a, (list, np.ndarray)) and len(a) > 0 else 0.0
            for a in g[speed_array_col]
        ])

        odo_diff_vals = pd.to_numeric(g[odo_col], errors="coerce").diff().to_numpy()
        np.nan_to_num(odo_diff_vals, copy=False, nan=0.0)

        cond_max_speed = (max_speed_vals < max_speed_in_stop_kph)
        cond_pct_stationary = (pct_stationary_vals >= stationary_sample_pct_thr)
        cond_odo_diff = (np.abs(odo_diff_vals) < thr_dist_km)
        
        if debug:
            condition_sums = cond_max_speed.astype(int) + cond_pct_stationary.astype(int) + cond_odo_diff.astype(int)
            mismatch_mask = (condition_sums == 1) | (condition_sums == 2)
            
            if np.any(mismatch_mask):
                mismatch_indices = g.index[mismatch_mask]
                reportable_mismatches = [
                    idx for idx in mismatch_indices 
                    if (idx > 0) or (cond_max_speed[idx] != cond_pct_stationary[idx])
                ]

                if reportable_mismatches:
                    print(f"\n[DEBUG] Mismatches found in trip_id: {trip_id}")
                    for idx in reportable_mismatches:
                        odo_note = "(first row, diff is artifact)" if idx == 0 else ""
                        print(
                            f"  - Row index {idx}: "
                            f"max_speed_ok={cond_max_speed[idx]} (val={max_speed_vals[idx]:.2f}), "
                            f"pct_stationary_ok={cond_pct_stationary[idx]} (val={pct_stationary_vals[idx]:.2f}), "
                            f"odo_diff_ok={cond_odo_diff[idx]} (val={odo_diff_vals[idx]:.4f}) {odo_note}"
                        )

        is_stop = cond_max_speed & cond_pct_stationary & cond_odo_diff
        g["stop_flag"] = np.where(is_stop, stop_indicator, 0)
        processed_trips.append(g)

    print(f" - Stop detection complete.")
    df_result = pd.concat(processed_trips, ignore_index=True)

    # --- Add summary statistics ---
    print("\n--- Natural Stop Statistics ---")
    total_rows = len(df_result)
    total_natural_stops = (df_result["stop_flag"] == stop_indicator).sum()
    total_trips = df_result[trip_id_col].nunique()
    trips_with_stops = df_result[df_result["stop_flag"] == stop_indicator][trip_id_col].nunique()
    avg_stops_per_trip = total_natural_stops / total_trips if total_trips > 0 else 0

    print(f" - Total rows processed: {total_rows:,}")
    print(f" - Total natural stops detected: {total_natural_stops:,} ({total_natural_stops / total_rows:.2%})")
    print(f" - Total unique trips: {total_trips:,}")
    print(f" - Trips containing at least one stop: {trips_with_stops:,} ({trips_with_stops / total_trips:.2%})")
    print(f" - Average natural stops per trip (overall): {avg_stops_per_trip:.2f}")
    
    return df_result


def incorporate_gap_flags(df, gap_flag_col='is_start_after_gap', stop_flag_col='stop_flag',
                          trip_id_col='trip_id', stop_indicator=-1, 
                          gap_indicator=-2, stop_and_gap_indicator=-3):
    """
    Updates the 'stop_flag' column to incorporate time gap information and prints a summary.
    Operates directly on the DataFrame.
    """
    print("\n--- Incorporating Time Gap Flags into Segmentation ---")

    if gap_flag_col not in df.columns:
        warnings.warn(f"Warning: Gap flag column '{gap_flag_col}' not found. Skipping this step.")
        return df

    # --- Capture initial state for summary ---
    total_rows = len(df)
    total_trips = df[trip_id_col].nunique()
    initial_natural_stops = (df[stop_flag_col] == stop_indicator).sum()
    initial_gaps = (df[gap_flag_col] == 1).sum()

    # --- Core Logic ---
    stop_and_gap_mask = (df[gap_flag_col] == 1) & (df[stop_flag_col] == stop_indicator)
    gap_only_mask = (df[gap_flag_col] == 1) & (df[stop_flag_col] != stop_indicator)

    df.loc[stop_and_gap_mask, stop_flag_col] = stop_and_gap_indicator
    df.loc[gap_only_mask, stop_flag_col] = gap_indicator
    
    # --- Capture final state for summary ---
    final_stop_and_gap = (df[stop_flag_col] == stop_and_gap_indicator).sum()
    final_gap_only = (df[stop_flag_col] == gap_indicator).sum()
    final_natural_stops = (df[stop_flag_col] == stop_indicator).sum()
    
    trips_with_any_gap_flag = df[df[stop_flag_col].isin([gap_indicator, stop_and_gap_indicator])][trip_id_col].nunique()

    # --- Print Summary ---
    print("\n--- Gap Flag Incorporation Summary ---")
    print(f" - Total rows processed: {total_rows:,}")
    print(f" - Total unique trips: {total_trips:,}")
    print(f" - Initial 'is_start_after_gap' flags found: {initial_gaps:,}")
    print(f" - Trips affected by gaps: {trips_with_any_gap_flag:,} ({trips_with_any_gap_flag/total_trips:.2%})")
    print("-" * 20)
    print("Breakdown of New Gap Flags:")
    print(f" - Rows marked as 'Gap Only' [-2]: {final_gap_only:,} ({final_gap_only/total_rows:.2%})")
    print(f"   (Gaps that occurred during driving)")
    print(f" - Rows marked as 'Stop & Gap' [-3]: {final_stop_and_gap:,} ({final_stop_and_gap/total_rows:.2%})")
    print(f"   (Gaps that coincided with a natural stop)")
    print("-" * 20)
    print("Impact on Natural Stops:")
    print(f" - Initial natural stops: {initial_natural_stops:,}")
    print(f" - Natural stops converted to 'Stop & Gap': {final_stop_and_gap:,}")
    print(f" - Remaining 'Natural Stop Only' [-1]: {final_natural_stops:,}")
    
    return df

# ==============================================================================
# --- New, Rule-Based Segmentation Functions (V5 - Constructive Logic) ---
# ==============================================================================

def find_valid_driving_blocks_v5(flags, min_win, full_stop):
    """
    (Unchanged from v4) Scans the trip's flags to find continuous blocks of driving
    that adhere to the strict start/end/content rules.
    """
    blocks = []
    i = 0
    n_samples = len(flags)
    while i < n_samples:
        if flags[i] not in [0, -2]:
            i += 1
            continue
        block_start = i
        consecutive_stops = 0
        j = block_start
        while j < n_samples:
            flag = flags[j]
            if j > block_start and flag in [-2, -3]: break
            if flag == -1: consecutive_stops += 1
            else: consecutive_stops = 0
            if consecutive_stops >= full_stop:
                j -= (full_stop - 1)
                break
            j += 1
        potential_end = j
        block_end = potential_end
        while block_end > block_start and flags[block_end - 1] != 0:
            block_end -= 1
        driving_samples_in_block = np.sum(np.isin(flags[block_start:block_end], [0, -2]))
        if driving_samples_in_block >= min_win:
            blocks.append((block_start, block_end))
        i = potential_end
    return blocks


def partition_and_label_block_v5(segment_labels, flags, block_start, block_end, 
                                 min_win, random_state, start_segment_id):
    """
    Uses a constructive approach to build segments sample-by-sample,
    ensuring all rules are met without complex post-correction. This is the
    most robust and logically sound version.
    """
    block_flags = flags[block_start:block_end]
    driving_samples_in_block = np.sum(np.isin(block_flags, [0, -2]))
    
    num_segments = driving_samples_in_block // min_win
    if num_segments == 0:
        return start_segment_id

    remainder = driving_samples_in_block % min_win
    try:
        partition_additions = random_partition(remainder, num_segments, random_state)
    except Exception:
        base_add = remainder // num_segments
        extra_add = remainder % num_segments
        partition_additions = [base_add + 1] * extra_add + [base_add] * (num_segments - extra_add)

    target_lengths = [min_win + add for add in partition_additions]
    
    current_segment_id = start_segment_id
    cursor_in_block = 0

    for target_len in target_lengths:
        # 1. Find a valid start for the segment (skip leading stops)
        segment_start_cursor = cursor_in_block
        while segment_start_cursor < len(block_flags) and block_flags[segment_start_cursor] == -1:
            segment_start_cursor += 1
        
        if segment_start_cursor >= len(block_flags):
            break # No more data in block to start a segment

        # 2. Build the segment constructively
        driving_samples_counted = 0
        segment_end_cursor = segment_start_cursor
        
        while segment_end_cursor < len(block_flags) and driving_samples_counted < target_len:
            flag = block_flags[segment_end_cursor]
            if flag in [0, -2]:
                driving_samples_counted += 1
            segment_end_cursor += 1
        
        # 3. Validate the constructed segment
        # Check if we found enough driving samples (could fail if block ends early)
        if driving_samples_counted >= min_win:
            # The segment is valid by construction.
            # The last driving sample increment guarantees it ends correctly.
            start_idx_global = block_start + segment_start_cursor
            end_idx_global = block_start + segment_end_cursor
            
            # Label only the driving points (flag == 0) within the valid segment range,
            # preserving the original non-zero flags (-1, -2, -3).
            segment_range_indices = np.arange(start_idx_global, end_idx_global)
            # Identify which of the original flags in this range were 0
            driving_points_mask = (flags[segment_range_indices] == 0)
            # Apply the new segment ID only to those points
            segment_labels[segment_range_indices[driving_points_mask]] = current_segment_id
            
            current_segment_id += 1
        
        # 4. Advance the main cursor to the end of the last attempted segment
        cursor_in_block = segment_end_cursor

    return current_segment_id


def split_rem_non_overlap_walk_forward_seg(df, trip_id_col='trip_id', stop_flag_col='stop_flag',
                                              min_win=10, full_stop=3, random_state=42):
    """
    Assigns segment IDs based on driving periods, using the V5 constructive logic
    for the most robust and correct rule enforcement.
    """
    print("\n--- Splitting Trips into Segments (V5 - Constructive Logic) ---")
    print(f" - Minimum driving rows per segment (min_win): {min_win}")
    print(f" - Consecutive non-driving events indicating segment end (full_stop): {full_stop}")

    trips_seg_dict = {}
    total_segments_created = 0

    for trip_id, trip_df in df.groupby(trip_id_col, sort=False):
        flags = trip_df[stop_flag_col].values
        segment_labels = flags.copy().astype(int)
        current_segment_id = 1
        
        valid_blocks = find_valid_driving_blocks_v5(flags, min_win, full_stop)
        
        for block_start, block_end in valid_blocks:
            next_id = partition_and_label_block_v5(
                segment_labels,
                flags,
                block_start,
                block_end,
                min_win,
                random_state,
                current_segment_id
            )
            current_segment_id = next_id
        
        trips_seg_dict[trip_id] = segment_labels
        total_segments_created += (current_segment_id - 1)

    print(f" - Segmentation complete for {len(df[trip_id_col].unique()):,} trips.")
    print(f" - Total segments created: {total_segments_created:,}")
    return trips_seg_dict

def evaluate_segmentation_rules(df, trips_seg_dict, stop_flag_col='stop_flag', 
                                trip_id_col='trip_id', min_win=10, full_stop=3, 
                                max_print=10):
    """
    Evaluates the segmentation output against a comprehensive set of logical rules.
    This version uses a robust method to define the "segment span" based on the
    explicit rules provided.

    Rules Checked (as per user specification):
    1.  **Start Rule:** A segment's span must start with a driving sample (0) or a gap-start (-2).
        - Valid starts: `[0,...]`, `[-2, 0,...]`, `[-2, -1, 0,...]`
        - Invalid starts: `[-1, 0,...]`, `[-3, 0,...]`
    2.  **End Rule:** A segment's span must end with a driving sample (0).
    3.  **Mid-Segment Rule:** The middle part of a segment cannot contain a gap flag (-2 or -3).
    4.  **Min Length Rule:** Must contain at least `min_win` driving samples (flags 0 or -2).
    5.  **Max Length Rule:** Must contain no more than `2 * min_win - 1` driving samples.
    6.  **Consecutive Stops Rule:** Cannot contain `full_stop` or more consecutive stop flags (-1).
    """
    print("\n--- Evaluating Adherence to All Segmentation Rules ---")
    print(f" - Evaluation params: min_win={min_win}, full_stop={full_stop}")
    
    violations = {}
    
    df_flags = df.set_index(trip_id_col)[stop_flag_col]

    for trip_id, trip_labels in trips_seg_dict.items():
        trip_violations = []
        trip_flags = df_flags.loc[trip_id].values
        
        segment_ids = np.unique(trip_labels[trip_labels > 0])
        
        for seg_id in segment_ids:
            labeled_indices = np.where(trip_labels == seg_id)[0]
            if len(labeled_indices) == 0: continue
            
            first_labeled_idx = labeled_indices[0]
            last_labeled_idx = labeled_indices[-1]

            segment_start_idx = find_seg_start_idx(trip_labels, first_labeled_idx)

            # Define the final, correct segment span.
            segment_span_indices = np.arange(segment_start_idx, last_labeled_idx + 1)
            segment_span_flags = trip_flags[segment_span_indices]
            segment_span_labels = trip_labels[segment_span_indices]
            
            if len(segment_span_flags) == 0: continue

            # --- Apply Rules to the Correctly Identified Span ---

            # Rule 1: First Sample Check
            if segment_span_labels[0] in [-1, -3]:
                trip_violations.append(f"Segment {seg_id} violates START rule: Span starts with flag {segment_span_labels[0]} at index {segment_start_idx}.")

            # Rule 2: Last Sample Check
            if segment_span_labels[-1] in [-1, -2, -3]:
                trip_violations.append(f"Segment {seg_id} violates END rule: Ends with flag {segment_span_labels[-1]} at index {last_labeled_idx}.")

            # Rule 3: Mid-Segment Gap Check
            if len(segment_span_labels) > 2:
                mid_labels = segment_span_labels[1:-1]
                if np.any(np.isin(mid_labels, [-2, -3])):
                    violating_idx_local = 1 + np.where(np.isin(mid_labels, [-2, -3]))[0][0]
                    violating_idx_global = segment_span_labels[violating_idx_local]
                    trip_violations.append(f"Segment {seg_id} violates MID-SEGMENT rule: Contains gap flag {trip_labels[violating_idx_global]} at index {violating_idx_global}.")

            # Rules 4 & 5: Length Checks
            driving_samples_count = np.sum(np.isin(segment_span_labels, [seg_id, -2]))
            max_len = 2 * min_win - 1
            if driving_samples_count < min_win:
                trip_violations.append(f"Segment {seg_id} violates MIN LENGTH rule: Has {driving_samples_count} driving samples, less than min_win={min_win}.")
            if driving_samples_count > max_len:
                trip_violations.append(f"Segment {seg_id} violates MAX LENGTH rule: Has {driving_samples_count} driving samples, more than max_allowed={max_len}.")

            # Rule 6: Consecutive Stops Check
            max_consecutive_stops = 0
            current_consecutive_stops = 0
            for flag in segment_span_labels:
                if flag == -1:
                    current_consecutive_stops += 1
                else:
                    max_consecutive_stops = max(max_consecutive_stops, current_consecutive_stops)
                    current_consecutive_stops = 0
            max_consecutive_stops = max(max_consecutive_stops, current_consecutive_stops)
            
            if max_consecutive_stops >= full_stop:
                trip_violations.append(f"Segment {seg_id} violates CONSECUTIVE STOP rule: Found a sequence of {max_consecutive_stops} stops, >= full_stop={full_stop}.")

        if trip_violations:
            violations[trip_id] = trip_violations
            
    # --- Reporting Phase ---
    if not violations:
        print("✅ Success! All segments adhere to the defined rules.")
    else:
        print(f"❌ Found violations in {len(violations)} trips:")
        for i, (trip_id, viol_list) in enumerate(violations.items()):
            if i >= max_print:
                print(f"\n... and {len(violations) - max_print} more trips with violations.")
                break
            
            print(f"\n--- Violations in Trip ID: {trip_id} ---")
            for v in viol_list:
                print(f"  - {v}")
            
            print("\n  Full trip data for diagnostics:")
            trip_labels = trips_seg_dict[trip_id]
            trip_flags = df_flags.loc[trip_id].values
            comparison_df = pd.DataFrame({'Segment_Label': trip_labels, 'Stop_Flag': trip_flags})
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 100):
                print(comparison_df)
            print("-" * (len(str(trip_id)) + 28))

    return violations

def summarize_segmentation_results(df, trips_seg_dict, trip_id_col='trip_id', stop_flag_col='stop_flag',
                                    full_stop=3, max_print_containing=3):
    """
    Analyzes the segmentation output and prints a detailed summary.
    This function is now 100% consistent with the evaluation logic, defining
    segment length by its full, valid span.
    """
    print("\n" + "="*50)
    print("--- Detailed Segmentation Results Summary (Rule-Consistent) ---")
    print("="*50)

    if not trips_seg_dict:
        print("The segmentation dictionary is empty. No results to summarize.")
        return

    # --- Data Preparation ---
    num_total_trips = len(trips_seg_dict)
    all_labels_flat = np.concatenate(list(trips_seg_dict.values()))
    df_flags = df.set_index(trip_id_col)[stop_flag_col]
    
    segments_per_trip = [np.max(labels) if np.any(labels > 0) else 0 for labels in trips_seg_dict.values()]
    segments_per_trip_ser = pd.Series(segments_per_trip)

    # --- 1. High-Level Trip & Segment Counts ---
    print("\n--- I. Overall Trip & Segment Statistics ---")
    total_segments = int(segments_per_trip_ser.sum())
    trips_with_segments = segments_per_trip_ser[segments_per_trip_ser > 0]
    num_trips_with_segments = len(trips_with_segments)
    num_trips_without_segments = num_total_trips - num_trips_with_segments

    print(f" - Total trips processed: {num_total_trips:,}")
    print(f" - Total segments created: {total_segments:,}")
    print(f" - Trips with NO segments (unsegmented): {num_trips_without_segments:,} ({num_trips_without_segments/num_total_trips:.2%})")
    print(f" - Trips containing at least one segment: {num_trips_with_segments:,} ({num_trips_with_segments/num_total_trips:.2%})")

    for i in range(1, max_print_containing):
        count_i = (segments_per_trip_ser > i).sum()   # threshold = i+1 segments
        pct_i   = count_i / num_total_trips
        print(f" - Trips containing at least {i+1} segments: {count_i:,} ({pct_i:.2%})")

    # --- 2. Segments Per Trip Distribution ---
    print("\n--- II. Segments-Per-Trip Distribution (for segmented trips) ---")
    if num_trips_with_segments > 0:
        print(f" - Average segments per trip (overall): {segments_per_trip_ser.mean():.2f}")
        print(f" - Average segments per trip (segmented only): {trips_with_segments.mean():.2f}")
        print(f" - Trip with maximum segments: {int(trips_with_segments.max())} segments")
        print("\nDistribution of segment counts:")
        print(trips_with_segments.describe().to_string())
    else:
        print("No trips were segmented.")

    # --- 3. Row-Level Label Breakdown ---
    print("\n--- III. Final Row-Level Label Breakdown ---")
    total_rows = len(all_labels_flat)
    segmented_rows = (all_labels_flat > 0).sum()
    unsegmented_driving_rows = (all_labels_flat == 0).sum()
    stop_rows = (all_labels_flat == -1).sum()
    gap_only_rows = (all_labels_flat == -2).sum()
    stop_and_gap_rows = (all_labels_flat == -3).sum()
    
    print(f"Total rows across all trips: {total_rows:,}\n")
    print(f" - Rows assigned to a driving segment (>0): {segmented_rows:>10,} ({segmented_rows/total_rows:8.2%})")
    print(f" - Rows of unsegmented driving data (0): {unsegmented_driving_rows:>10,} ({unsegmented_driving_rows/total_rows:8.2%})")
    print(f" - Rows marked as 'Natural Stop Only' (-1): {stop_rows:>10,} ({stop_rows/total_rows:8.2%})")
    print(f" - Rows marked as 'Gap Only' (-2): {gap_only_rows:>10,} ({gap_only_rows/total_rows:8.2%})")
    print(f" - Rows marked as 'Stop & Gap' (-3): {stop_and_gap_rows:>10,} ({stop_and_gap_rows/total_rows:8.2%})")

    # --- 4. Segment Length Distribution (Corrected Logic) ---
    print("\n--- IV. Segment Span Length (in rows) Distribution ---")
    if total_segments > 0:
        all_segment_span_lengths = []
        # Iterate through each trip to calculate segment span lengths individually.
        for trip_id, labels in trips_seg_dict.items():
            trip_flags = df_flags.loc[trip_id].values
            segment_ids = np.unique(labels[labels > 0])
            
            for seg_id in segment_ids:
                labeled_indices = np.where(labels == seg_id)[0]
                if len(labeled_indices) == 0: continue
                
                first_labeled_idx = labeled_indices[0]
                last_labeled_idx = labeled_indices[-1]

                # Use the same "Last Anchor" logic as the evaluation function
                search_space_start_idx = 0
                for i in range(first_labeled_idx - 1, -1, -1):
                    if i >= full_stop - 1 and np.all(trip_flags[i - (full_stop - 1) : i + 1] == -1):
                        search_space_start_idx = i + 1; break
                    if labels[i] > 0 or trip_flags[i] in [0, -3]:
                        search_space_start_idx = i + 1; break
                    search_space_start_idx = i
                
                anchor_idx = -1
                for i in range(first_labeled_idx - 1, search_space_start_idx - 1, -1):
                    if trip_flags[i] == -2: anchor_idx = i; break
                
                segment_start_idx = anchor_idx if anchor_idx != -1 else first_labeled_idx
                
                span_length = (last_labeled_idx - segment_start_idx) + 1
                all_segment_span_lengths.append(span_length)
        
        segment_lengths_ser = pd.Series(all_segment_span_lengths)
        print("Distribution of segment span lengths (in rows):")
        print(segment_lengths_ser.describe().to_string())
    else:
        print("No segments were created to analyze length.")
        
    print("\n" + "="*50)
    print("--- End of Summary ---")
    print("="*50)

def find_and_print_trip_with_max_segment(df, trips_seg_dict, trip_id_col='trip_id', stop_flag_col='stop_flag', full_stop=3):
    """
    Finds the trip containing the single longest segment by its full span length
    and prints its details. This function is now 100% consistent with the
    evaluation logic.
    """
    print("\n--- Finding Trip with Maximum Segment Span Length ---")

    if not trips_seg_dict:
        print("Segmentation dictionary is empty.")
        return

    max_span_len_found = 0
    trip_id_with_max_len = None
    seg_id_with_max_len = None
    
    df_flags = df.set_index(trip_id_col)[stop_flag_col]

    # Iterate through each trip to find the segment with the longest span
    for trip_id, labels in trips_seg_dict.items():
        trip_flags = df_flags.loc[trip_id].values
        segment_ids = np.unique(labels[labels > 0])

        for seg_id in segment_ids:
            labeled_indices = np.where(labels == seg_id)[0]
            if len(labeled_indices) == 0: continue
            
            first_labeled_idx = labeled_indices[0]
            last_labeled_idx = labeled_indices[-1]

            # Use the same "Last Anchor" logic as the evaluation function
            search_space_start_idx = 0
            for i in range(first_labeled_idx - 1, -1, -1):
                if i >= full_stop - 1 and np.all(trip_flags[i - (full_stop - 1) : i + 1] == -1):
                    search_space_start_idx = i + 1; break
                if labels[i] > 0 or trip_flags[i] in [0, -3]:
                    search_space_start_idx = i + 1; break
                search_space_start_idx = i
            
            anchor_idx = -1
            for i in range(first_labeled_idx - 1, search_space_start_idx - 1, -1):
                if trip_flags[i] == -2: anchor_idx = i; break
            
            segment_start_idx = anchor_idx if anchor_idx != -1 else first_labeled_idx
            
            current_span_len = (last_labeled_idx - segment_start_idx) + 1
            
            if current_span_len > max_span_len_found:
                max_span_len_found = current_span_len
                trip_id_with_max_len = trip_id
                seg_id_with_max_len = seg_id

    if trip_id_with_max_len is None:
        print("No trips with segments were found.")
        return

    # --- Reporting Phase ---
    print(f"\nTrip with the longest segment span: '{trip_id_with_max_len}'")
    print(f"Segment ID with the longest span: {seg_id_with_max_len}")
    print(f"Length of its span: {max_span_len_found} rows")

    # Get the specific trip's data from the DataFrame
    trip_data = df[df[trip_id_col] == trip_id_with_max_len]
    
    # Get the stop flags and segment labels for this trip
    stop_flags_for_trip = trip_data[stop_flag_col].values
    segment_labels_for_trip = trips_seg_dict[trip_id_with_max_len]

    # Create a DataFrame for easy side-by-side comparison
    comparison_df = pd.DataFrame({
        'Segment_Label': segment_labels_for_trip,
        'Stop_Flag': stop_flags_for_trip
    })

    print("\nSegment labels and stop flags for this trip (side-by-side):")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(comparison_df)

def print_examples(df, trips_seg_dict, trip_id_col='trip_id', stop_flag_col='stop_flag', segments_to_print=5):
    """
    Print n examples of stop columns and corresponding segments.
    """
    for i,(k,v) in enumerate(trips_seg_dict.items()):
        stop_flags_for_trip = df[df[trip_id_col] == k][stop_flag_col].values
        segment_labels_for_trip = trips_seg_dict[k]

        # Create a DataFrame for easy side-by-side comparison
        comparison_df = pd.DataFrame({
            'Segment_Label': segment_labels_for_trip,
            'Stop_Flag': stop_flags_for_trip
        })

        print(f"\nSegment labels and stop flags for trip {k}:")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(comparison_df)

        if i == segments_to_print:
            break

def save_seg_dist_plots(df, trips_seg_dict, trip_id_col='trip_id', stop_flag_col='stop_flag', full_stop=3,
                        output_dir=r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segmentation_dict"):
    """
    Saves aesthetic plots for distribution of number of segments per trip and number of minutes (samples) per segment.
    Uses the script's output directory for saving the plots.
    """ 

    # Collect data for plots
    num_segments_per_trip = []
    segment_lengths = []

    for trip_id, trip_labels in trips_seg_dict.items():
        trip_df = df[df[trip_id_col] == trip_id].copy()
        if len(trip_df) != len(trip_labels):
            warnings.warn(f"Length mismatch for trip_id {trip_id}. Skipping trip.")
            continue

        unique_seg_ids = np.unique(trip_labels[trip_labels > 0]).astype(int)
        if len(unique_seg_ids) == 0: 
            warnings.warn(f'{trip_id} Trip skipped: No valid positive segment IDs found.')
            continue 

        num_segments_per_trip.append(len(unique_seg_ids))   
        
        for seg_id in unique_seg_ids:

            labeled_indices = np.where(trip_labels == seg_id)[0]
            if len(labeled_indices) == 0: 
                warnings.warn(f"No samples for trip_id {trip_id} seg {seg_id}!")
                continue
            
            first_labeled_idx = labeled_indices[0]
            last_labeled_idx = labeled_indices[-1]

            segment_start_idx = find_seg_start_idx(trip_labels, first_labeled_idx, full_stop=full_stop)
            segment_span_indices = np.arange(segment_start_idx, last_labeled_idx + 1)
            segment_length = len(segment_span_indices)
            segment_lengths.append(segment_length)

    # Set aesthetic style (using 'ggplot' as it's built-in to matplotlib)
    plt.style.use('ggplot')

    segments_per_trip_ser = pd.Series(num_segments_per_trip)
    segments_counts = segments_per_trip_ser.value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(segments_counts.index, segments_counts.values, color='skyblue', edgecolor='black', width=0.8)
    ax1.set_title('Distribution of Number of Segments per Trip', fontsize=16)
    ax1.set_xlabel('Number of Segments', fontsize=14)
    ax1.set_ylabel('Number of Trips', fontsize=14)
    ax1.set_xticks(segments_counts.index)
    ax1.tick_params(labelsize=12)
    
    # Add mean and median
    mean_segments = segments_per_trip_ser.mean()
    median_segments = segments_per_trip_ser.median()
    ax1.axvline(mean_segments, color='red', linestyle='--', label=f'Mean: {mean_segments:.2f}')
    ax1.axvline(median_segments, color='green', linestyle='-', label=f'Median: {median_segments:.2f}')
    ax1.legend(fontsize=14)
    
    # Set aesthetic style (using 'ggplot' as it's built-in to matplotlib)
    plt.style.use('ggplot')

    segments_per_trip_ser = pd.Series(num_segments_per_trip)
    segments_counts = segments_per_trip_ser.value_counts().sort_index()
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(segments_counts.index, segments_counts.values, color='skyblue', edgecolor='black', width=0.8)
    ax1.set_title('Distribution of Number of Segments per Trip', fontsize=16)
    ax1.set_xlabel('Number of Segments', fontsize=14)
    ax1.set_ylabel('Number of Trips', fontsize=14)
    ax1.set_xticks(segments_counts.index)
    ax1.tick_params(labelsize=12)
    
    # Add mean and median
    mean_segments = segments_per_trip_ser.mean()
    median_segments = segments_per_trip_ser.median()
    skew_segments = segments_per_trip_ser.skew()
    ax1.axvline(mean_segments, color='red', linestyle='--', label=f'Mean: {mean_segments:.2f}')
    ax1.axvline(median_segments, color='green', linestyle='-', label=f'Median: {median_segments:.2f}')
    ax1.plot([], [], ' ', label=f'Skewness: {skew_segments:.2f}')
    ax1.legend(loc='upper right', fontsize=14)
    
    plot1_path = os.path.join(output_dir, 'segments_per_trip_distribution.png')
    fig1.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # Plot 2: Distribution of segment lengths (in minutes/samples) (Bar plot with actual ticks)
    segment_lengths_ser = pd.Series(segment_lengths)
    lengths_counts = segment_lengths_ser.value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(lengths_counts.index, lengths_counts.values, color='lightgreen', edgecolor='black', width=0.8)
    ax2.set_title('Distribution of Segment Lengths', fontsize=16)
    ax2.set_xlabel('Length of Segment (Minutes)', fontsize=14)
    ax2.set_ylabel('Number of Segments', fontsize=14)
    ax2.set_xticks(lengths_counts.index)
    ax2.tick_params(labelsize=12)
    
    # Add mean and median
    mean_lengths = segment_lengths_ser.mean()
    median_lengths = segment_lengths_ser.median()
    skew_lengths = segment_lengths_ser.skew()
    ax2.axvline(mean_lengths, color='red', linestyle='--', label=f'Mean: {mean_lengths:.2f}')
    ax2.axvline(median_lengths, color='green', linestyle='-', label=f'Median: {median_lengths:.2f}')
    ax2.plot([], [], ' ', label=f'Skewness: {skew_lengths:.2f}')
    ax2.legend(loc='upper right', fontsize=14)
    
    plot2_path = os.path.join(output_dir, 'segment_lengths_distribution.png')
    fig2.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print(f"Plots saved to: {plot1_path} and {plot2_path}")

# --- Main Execution Logic ---
def main():
    """Loads cleaned data, finds stops, segments trips, and saves results."""

    # --- Configuration ---
    RANDOM_STATE = 42
    CLEANED_DATA_PATH = r'C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\clean_df\clean_df.parquet'
    SEGMENTATION_DICT_DIR = r"C:\Users\goldy\OneDrive\Documents\Y-DATA 24-25\Stellantis Project\End-to-End\segmentation_dict"
    SEGMENTATION_DICT_FILENAME = 'trips_seg_dict_v7'
    OUTPUT_FORMAT = 'pickle'
    
    MIN_SEGMENT_POINTS = 10
    CONSECUTIVE_STOPS_FOR_BREAK = 3
    # --------------------

    # --- Standard Column Names ---
    TRIP_ID_COL = 'trip_id'
    TIME_COL = 'timestamp'
    ODO_COL = 'current_odo'
    SPEED_ARRAY_COL = 'speed_array'
    GAP_FLAG_COL = 'is_start_after_gap'
    STOP_FLAG_COL = 'stop_flag'
    # --------------------

    print(f"--- Starting Trip Segmentation Pipeline ---")
    print(f"\nInput cleaned data file: {CLEANED_DATA_PATH}")
    print(f"Output directory for segmentation dict: {SEGMENTATION_DICT_DIR}")

    try:
        # 1. Load Cleaned Data
        print("\n--- Loading Cleaned Data ---")
        df_clean = load_file(CLEANED_DATA_PATH)
        if not isinstance(df_clean, pd.DataFrame): raise TypeError("Loaded data is not a DataFrame.")
        print(f"Loaded cleaned data with shape: {df_clean.shape}")

        # 2. Ensure Sorting
        print(f"\n--- Sorting data by {TRIP_ID_COL}, {TIME_COL} ---")
        df_clean = sort_df_by_trip_time(df_clean, trip_id_col=TRIP_ID_COL, time_col=TIME_COL)
        print("Data sorting complete.")

        # 3. Find Natural Stops using the new high-performance, vectorized function
        # This now returns a DataFrame with a 'stop_flag' column.
        # To debug stop detection, set debug=True
        df_clean = find_natural_stops(
            df_clean,
            trip_id_col=TRIP_ID_COL,
            speed_array_col=SPEED_ARRAY_COL,
            odo_col=ODO_COL,
            debug=False  # Set to True to print mismatch details
        )

        # 4. Incorporate Gap Flags directly into the DataFrame
        df_clean = incorporate_gap_flags(
            df=df_clean,
            gap_flag_col=GAP_FLAG_COL,
            stop_flag_col=STOP_FLAG_COL,
            trip_id_col=TRIP_ID_COL 
        )

        # 5. Segment Trips using the DataFrame's flag column
        trips_seg_dict = split_rem_non_overlap_walk_forward_seg(
            df_clean,
            trip_id_col=TRIP_ID_COL,
            stop_flag_col=STOP_FLAG_COL,
            min_win=MIN_SEGMENT_POINTS,
            full_stop=CONSECUTIVE_STOPS_FOR_BREAK,
            random_state=RANDOM_STATE
        )

        # 6. Evaluate the segmentation with the new comprehensive rules
        evaluate_segmentation_rules(
            df=df_clean,
            trips_seg_dict=trips_seg_dict,
            stop_flag_col=STOP_FLAG_COL,
            trip_id_col=TRIP_ID_COL,
            min_win=MIN_SEGMENT_POINTS,
            full_stop=CONSECUTIVE_STOPS_FOR_BREAK,
            max_print=1000
        )

        # 7. Summarize results using the same rule-consistent logic
        summarize_segmentation_results(
            df=df_clean,
            trips_seg_dict=trips_seg_dict,
            trip_id_col=TRIP_ID_COL,
            stop_flag_col=STOP_FLAG_COL,
            full_stop=CONSECUTIVE_STOPS_FOR_BREAK
        )
        
        # 8. Find the max segment using the same rule-consistent logic
        find_and_print_trip_with_max_segment(
            df=df_clean,
            trips_seg_dict=trips_seg_dict,
            trip_id_col=TRIP_ID_COL,
            stop_flag_col=STOP_FLAG_COL,
            full_stop=CONSECUTIVE_STOPS_FOR_BREAK
        )

        print_examples(
            df=df_clean,
            trips_seg_dict=trips_seg_dict,
            trip_id_col=TRIP_ID_COL,
            stop_flag_col=STOP_FLAG_COL,
            segments_to_print=15
        )

        save_seg_dist_plots(
            df=df_clean, 
            trips_seg_dict=trips_seg_dict, 
            trip_id_col=TRIP_ID_COL, 
            stop_flag_col=STOP_FLAG_COL, 
            full_stop=CONSECUTIVE_STOPS_FOR_BREAK,
            output_dir=SEGMENTATION_DICT_DIR
            )

        # 6. Save the final segmentation dictionary
        print(f"\n--- Saving Segmentation Dictionary ---")
        save_file(
            data=trips_seg_dict, path=SEGMENTATION_DICT_DIR,
            file_name=SEGMENTATION_DICT_FILENAME, format=OUTPUT_FORMAT
        )
        
        print("\n--- Trip Segmentation Pipeline Complete ---")

    except (FileNotFoundError, TypeError, ValueError, KeyError, ImportError, Exception) as e:
        print(f"\n--- Segmentation Pipeline Failed ---")
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    main()