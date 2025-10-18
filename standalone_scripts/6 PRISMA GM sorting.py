import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import tkinter.font as tkFont
import pandas as pd
import numpy as np
from scipy.stats import gmean  # For geometric mean


class SelectiveDataProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Individual Alloy Data Processor")
        self.root.geometry("1600x850")  # Adjusted for better layout
        self.df_full = None
        self.current_results_df = None
        # Updated input identifier column as per user's latest request
        self.input_identifier_cols = ['Composition']

        # Updated base configuration for objectives to match new Excel column names
        base_objectives_config = [
            {'name': 'Time [s]', 'goal': 'minimize'},
            {'name': 'Mean radius (Nb-V)C', 'goal': 'minimize'},
            {'name': 'Number density (Nb-V)C', 'goal': 'maximize'},
            {'name': 'Volume fraction (Nb-V)C', 'goal': 'maximize'},
            {'name': 'Matrix composition Mo', 'goal': 'maximize'},
            {'name': 'Matrix composition C', 'goal': 'maximize'}
        ]

        self.objective_configs = []
        for oc in base_objectives_config:
            # Sanitize name for internal use (e.g., normalized column name)
            # Keeps original name for display and Excel lookup
            sanitized_name = ''.join(c if c.isalnum() or c == '_' else '_' for c in oc['name'])
            self.objective_configs.append({
                'name': oc['name'],  # This is the actual Excel column name
                'goal': oc['goal'],
                'var': tk.BooleanVar(value=True),
                'norm_col_name': f"{sanitized_name}_norm"  # Internal name for the normalized column
            })

        # These are the columns the program expects to find in the Excel file
        self.required_cols = self.input_identifier_cols + [oc['name'] for oc in self.objective_configs]
        self.last_sorted_column = 'Overall_Score'  # Default sort
        self.last_sort_order_asc = False  # Default sort order (descending for score)

        # --- UI Elements ---
        self.top_frame = ttk.Frame(root)
        self.top_frame.pack(padx=10, pady=5, fill="x")

        self.file_frame = ttk.LabelFrame(self.top_frame, text="Input File")
        self.file_frame.pack(side=tk.LEFT, padx=5, pady=5, fill="x", expand=True)

        self.load_button = ttk.Button(self.file_frame, text="Load Excel File", command=self.load_excel_file)
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.file_label = ttk.Label(self.file_frame, text="No file selected", width=40)
        self.file_label.pack(side=tk.LEFT, padx=5, pady=5)

        self.reprocess_button = ttk.Button(self.top_frame, text="Process Data", command=self.process_data_from_ui)
        self.reprocess_button.pack(side=tk.LEFT, padx=10, pady=10)
        self.reprocess_button.config(state=tk.DISABLED)

        self.export_button = ttk.Button(self.top_frame, text="Export to Excel", command=self.export_to_excel)
        self.export_button.pack(side=tk.LEFT, padx=5, pady=10)
        self.export_button.config(state=tk.DISABLED)

        # --- Debug Button ---
        self.debug_button = ttk.Button(self.top_frame, text="Debug Data", command=self.debug_data)
        self.debug_button.pack(side=tk.LEFT, padx=5, pady=10)
        self.debug_button.config(state=tk.DISABLED)

        # --- Info Frame ---
        self.info_frame = ttk.LabelFrame(root, text="Processing Info")
        self.info_frame.pack(padx=10, pady=5, fill="x")

        info_label_text = ("This tool processes data based on the objectives listed below.\n"
                           "Normalization and ranking is done SEPARATELY for each individual alloy composition.\n"
                           "Select which objectives contribute to the 'Overall_Score' (Geometric Mean).")
        ttk.Label(self.info_frame, text=info_label_text, justify=tk.LEFT).pack(padx=5, pady=5, anchor="w")

        # --- Objective Selection Frame ---
        self.objective_selection_frame = ttk.LabelFrame(root, text="Select objectives for ranking")
        self.objective_selection_frame.pack(padx=10, pady=5, fill="x")

        max_cols_checkbox = 2  # Adjusted for potentially longer names, can be 3 if space allows
        current_col = 0
        current_row = 0
        checkbox_frame_inner = ttk.Frame(self.objective_selection_frame)
        checkbox_frame_inner.pack(fill="x", padx=5, pady=5)

        for i, oc_config in enumerate(self.objective_configs):
            cb = ttk.Checkbutton(checkbox_frame_inner, text=f"{oc_config['name']} ({oc_config['goal']})",
                                 variable=oc_config['var'])
            cb.grid(row=current_row, column=current_col, padx=5, pady=2, sticky="w")
            current_col += 1
            if current_col >= max_cols_checkbox:
                current_col = 0
                current_row += 1

        # --- Results Display Frame ---
        self.results_frame = ttk.LabelFrame(root, text="Processed Results (Click column header to sort)")
        self.results_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.tree = ttk.Treeview(self.results_frame)
        self.tree.pack(fill="both", expand=True, padx=5, pady=5)

        vsb = ttk.Scrollbar(self.tree, orient="vertical", command=self.tree.yview)
        vsb.pack(side='right', fill='y')
        self.tree.configure(yscrollcommand=vsb.set)

        hsb = ttk.Scrollbar(self.tree, orient="horizontal", command=self.tree.xview)
        hsb.pack(side='bottom', fill='x')
        self.tree.configure(xscrollcommand=hsb.set)

    def debug_data(self):
        """Debug function to display detailed information about the loaded data"""
        if self.df_full is None:
            messagebox.showwarning("No Data", "Please load an Excel file first.")
            return

        debug_window = tk.Toplevel(self.root)
        debug_window.title("Debug Information")
        debug_window.geometry("800x600")

        # Create text widget with scrollbar
        text_frame = ttk.Frame(debug_window)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)

        text_widget = tk.Text(text_frame, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)

        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Collect debug information
        debug_info = []
        debug_info.append("=== DEBUG INFORMATION ===\n")
        debug_info.append(f"DataFrame shape: {self.df_full.shape}")
        debug_info.append(f"Columns found in Excel: {list(self.df_full.columns)}\n")

        # Show alloy compositions found
        if 'Composition' in self.df_full.columns:
            unique_alloys = self.df_full['Composition'].unique()
            debug_info.append(f"Unique alloy compositions found: {len(unique_alloys)}")
            for alloy in unique_alloys:
                count = (self.df_full['Composition'] == alloy).sum()
                debug_info.append(f"  {alloy}: {count} rows")
            debug_info.append("")

        # Check each objective column
        for oc_config in self.objective_configs:
            col_name = oc_config['name']
            debug_info.append(f"\n--- Column: {col_name} ---")

            if col_name in self.df_full.columns:
                col_data = self.df_full[col_name]
                debug_info.append(f"Data type: {col_data.dtype}")
                debug_info.append(f"Non-null count: {col_data.count()}")
                debug_info.append(f"Null count: {col_data.isna().sum()}")

                # Show first few values
                debug_info.append("First 5 raw values:")
                for i, val in enumerate(col_data.head(5)):
                    debug_info.append(f"  [{i}]: {repr(val)} (type: {type(val)})")

                # Try numeric conversion
                try:
                    # Clean and convert to numeric
                    cleaned_data = col_data.astype(str).str.strip()
                    cleaned_data = cleaned_data.replace('', np.nan)
                    numeric_data = pd.to_numeric(cleaned_data, errors='coerce')

                    debug_info.append(f"After numeric conversion:")
                    debug_info.append(f"  Min: {numeric_data.min()}")
                    debug_info.append(f"  Max: {numeric_data.max()}")
                    debug_info.append(f"  Mean: {numeric_data.mean()}")
                    debug_info.append(f"  Non-zero count: {(numeric_data != 0).sum()}")
                    debug_info.append(f"  Zero count: {(numeric_data == 0).sum()}")
                    debug_info.append(f"  NaN after conversion: {numeric_data.isna().sum()}")

                    # Show first few converted values
                    debug_info.append("First 5 converted values:")
                    for i, val in enumerate(numeric_data.head(5)):
                        debug_info.append(f"  [{i}]: {val}")

                except Exception as e:
                    debug_info.append(f"Error in numeric conversion: {e}")
            else:
                debug_info.append("Column NOT FOUND in Excel file!")

        # Display debug information
        text_widget.insert(tk.END, "\n".join(debug_info))
        text_widget.config(state=tk.DISABLED)

        # Also print to console for easy copying
        print("\n".join(debug_info))

    def load_excel_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=(("Excel files", "*.xlsx *.xls"), ("All files", "*.*"))
        )
        if not file_path:
            return

        try:
            self.df_full = pd.read_excel(file_path)

            # DEBUG: Print basic information about loaded data
            print("DEBUG: Successfully loaded Excel file")
            print("DEBUG: Columns found in Excel file:", self.df_full.columns.tolist())
            print("DEBUG: DataFrame shape:", self.df_full.shape)

            self.file_label.config(text=file_path.split('/')[-1])

            # Check for all required columns (identifiers + objectives)
            missing_cols = [col for col in self.required_cols if col not in self.df_full.columns]
            if missing_cols:
                messagebox.showerror("Error: Missing Columns",
                                     f"The Excel file is missing the following required columns:\n"
                                     f"{', '.join(missing_cols)}\n\n"
                                     f"Available columns: {', '.join(self.df_full.columns)}\n\n"
                                     f"Please ensure column names match exactly (case-sensitive, no extra spaces).")
                self.df_full = None
                self.current_results_df = None
                self.file_label.config(text="No file selected (Error)")
                self.clear_treeview()
                self.reprocess_button.config(state=tk.DISABLED)
                self.export_button.config(state=tk.DISABLED)
                self.debug_button.config(state=tk.DISABLED)
                return

            # Enhanced data validation and cleaning for objective columns
            for oc_config in self.objective_configs:
                col_name = oc_config['name']  # This is the actual Excel column name
                if col_name in self.df_full.columns:  # Should always be true due to above check
                    try:
                        print(f"DEBUG: Processing column '{col_name}'")
                        original_data = self.df_full[col_name]
                        print(f"DEBUG: Original data type: {original_data.dtype}")
                        print(f"DEBUG: First 5 original values: {original_data.head().tolist()}")

                        # Enhanced cleaning process
                        if original_data.dtype == 'object':
                            # Handle string data that might contain numbers
                            cleaned_data = original_data.astype(str).str.strip()
                            cleaned_data = cleaned_data.replace('', np.nan)
                            cleaned_data = cleaned_data.replace('nan', np.nan)
                        else:
                            cleaned_data = original_data

                        # Convert to numeric with error handling
                        numeric_data = pd.to_numeric(cleaned_data, errors='coerce')

                        print(f"DEBUG: After conversion - Non-null count: {numeric_data.count()}")
                        print(f"DEBUG: After conversion - Min: {numeric_data.min()}, Max: {numeric_data.max()}")
                        print(f"DEBUG: After conversion - Zero values: {(numeric_data == 0).sum()}")

                        # Check if conversion resulted in all zeros or all NaNs
                        if numeric_data.isna().all():
                            print(f"WARNING: All values in column '{col_name}' became NaN after conversion!")
                            print(f"Original unique values: {original_data.unique()}")
                        elif (numeric_data == 0).all():
                            print(f"WARNING: All values in column '{col_name}' are zero!")

                        # Update the dataframe with cleaned numeric data
                        self.df_full[col_name] = numeric_data

                    except Exception as e:
                        error_msg = f"Error processing column '{col_name}': {str(e)}"
                        print(f"DEBUG: {error_msg}")
                        messagebox.showerror("Data Processing Error", error_msg)
                        self.df_full = None
                        self.current_results_df = None
                        self.file_label.config(text="No file selected (Error)")
                        self.clear_treeview()
                        self.reprocess_button.config(state=tk.DISABLED)
                        self.export_button.config(state=tk.DISABLED)
                        self.debug_button.config(state=tk.DISABLED)
                        return

            # Enable buttons after successful loading
            self.reprocess_button.config(state=tk.NORMAL)
            self.debug_button.config(state=tk.NORMAL)
            self.process_data_from_ui()  # Automatically process after loading

        except Exception as e:
            error_msg = f"An unexpected error occurred while loading the file: {e}"
            print(f"DEBUG: {error_msg}")
            messagebox.showerror("Error Loading File", error_msg)
            self.df_full = None
            self.current_results_df = None
            self.file_label.config(text="No file selected (Error)")
            self.clear_treeview()
            self.reprocess_button.config(state=tk.DISABLED)
            self.export_button.config(state=tk.DISABLED)
            self.debug_button.config(state=tk.DISABLED)

    def process_data_from_ui(self):
        if self.df_full is None:
            messagebox.showwarning("No Data", "Please load an Excel file first.")
            return
        self.process_data()

    def process_data(self):
        if self.df_full is None:
            return

        df = self.df_full.copy()  # Work on a copy

        print("DEBUG: Starting individual alloy data processing...")

        # Get unique alloy compositions
        unique_alloys = df['Composition'].unique()
        print(f"DEBUG: Found {len(unique_alloys)} unique alloy compositions")

        # Initialize result dataframe list
        processed_alloy_dfs = []

        # Process each alloy separately
        for alloy in unique_alloys:
            print(f"\nDEBUG: Processing alloy: {alloy}")

            # Filter data for current alloy
            alloy_df = df[df['Composition'] == alloy].copy()
            print(f"DEBUG: Alloy {alloy} has {len(alloy_df)} rows")

            # --- 1. Calculate normalized objective columns for this alloy only ---
            for oc_config in self.objective_configs:
                col_name = oc_config['name']  # Actual Excel column name
                goal = oc_config['goal']
                norm_col_name = oc_config['norm_col_name']  # Internal normalized column name

                if col_name not in alloy_df.columns:  # Should not happen if load_excel_file worked
                    alloy_df[norm_col_name] = np.nan
                    continue

                values = alloy_df[col_name].astype(float)  # Should already be numeric from load_excel_file
                print(f"DEBUG: Normalizing '{col_name}' for alloy '{alloy}' with goal '{goal}'")
                print(f"DEBUG: Values range for {alloy} - Min: {values.min()}, Max: {values.max()}")

                min_val = values.min()
                max_val = values.max()

                if pd.isna(min_val) or pd.isna(max_val):  # Handle columns with all NaNs
                    print(f"DEBUG: Column '{col_name}' has all NaN values for alloy '{alloy}'")
                    alloy_df[norm_col_name] = np.nan
                    continue

                # Calculate relative difference to handle very small numbers
                relative_diff = abs(max_val - min_val) / max(abs(max_val), abs(min_val), 1e-15)

                if relative_diff < 1e-6:  # If relative difference is less than 0.0001%
                    print(
                        f"DEBUG: Column '{col_name}' has very small relative variation for alloy '{alloy}' ({relative_diff:.2e})")
                    print(f"DEBUG: Min: {min_val:.6e}, Max: {max_val:.6e}, Diff: {max_val - min_val:.6e}")
                    # For very small variations, use a more sensitive normalization
                    if goal == 'maximize':
                        alloy_df[norm_col_name] = (values - min_val) / (max_val - min_val)
                    elif goal == 'minimize':
                        alloy_df[norm_col_name] = (max_val - values) / (max_val - min_val)
                elif abs(max_val - min_val) < 1e-15:  # True constant values (absolute threshold)
                    print(
                        f"DEBUG: Column '{col_name}' has truly constant values for alloy '{alloy}', setting normalized to 1.0")
                    alloy_df[norm_col_name] = 1.0  # All values are effectively the same, max score
                elif goal == 'maximize':
                    alloy_df[norm_col_name] = (values - min_val) / (max_val - min_val)
                elif goal == 'minimize':
                    alloy_df[norm_col_name] = (max_val - values) / (max_val - min_val)

                alloy_df[norm_col_name] = alloy_df[norm_col_name].clip(0, 1)  # Ensure scores are between 0 and 1

                print(
                    f"DEBUG: Normalized '{col_name}' for {alloy} - Min: {alloy_df[norm_col_name].min()}, Max: {alloy_df[norm_col_name].max()}")

            # --- 2. Determine which normalized objectives to use for GMean based on checkboxes ---
            selected_normalized_cols_for_gmean = [
                oc['norm_col_name'] for oc in self.objective_configs if oc['var'].get()
            ]

            print(f"DEBUG: Selected columns for GMean for alloy '{alloy}': {selected_normalized_cols_for_gmean}")

            if not selected_normalized_cols_for_gmean:
                print(f"DEBUG: No objectives selected for GMean for alloy '{alloy}'")

            # --- 3. Calculate Geometric Mean (Overall Score) for this alloy ---
            gmean_values = []
            for _, row in alloy_df.iterrows():
                if not selected_normalized_cols_for_gmean:  # No objectives selected
                    current_gmean = 0.0
                else:
                    obj_values_for_gmean = [row[norm_col] for norm_col in selected_normalized_cols_for_gmean if
                                            pd.notna(row[norm_col])]
                    if not obj_values_for_gmean:  # All selected objectives for this row are NaN
                        current_gmean = 0.0
                    else:
                        current_gmean = gmean(obj_values_for_gmean)  # Handles 0s correctly
                gmean_values.append(current_gmean)

            alloy_df['Overall_Score'] = gmean_values
            alloy_df['Overall_Score'] = alloy_df['Overall_Score'].fillna(0)  # Fill NaN scores with 0

            print(
                f"DEBUG: Overall Score range for {alloy} - Min: {alloy_df['Overall_Score'].min()}, Max: {alloy_df['Overall_Score'].max()}")

            # Sort this alloy's data by Overall_Score (descending by default)
            alloy_df = alloy_df.sort_values(by='Overall_Score', ascending=False)

            # Add rank within this alloy (1 = best for this alloy)
            alloy_df['Rank_within_Alloy'] = range(1, len(alloy_df) + 1)

            # Add the processed alloy dataframe to our list
            processed_alloy_dfs.append(alloy_df)

        # Combine all processed alloys back together
        if processed_alloy_dfs:
            self.current_results_df = pd.concat(processed_alloy_dfs, ignore_index=True)

            # Sort by composition and then by rank within alloy for display
            self.current_results_df = self.current_results_df.sort_values(
                by=['Composition', 'Rank_within_Alloy'],
                ascending=[True, True]
            )

            print(f"DEBUG: Final combined dataframe shape: {self.current_results_df.shape}")
            self.export_button.config(state=tk.NORMAL)
        else:
            self.current_results_df = pd.DataFrame()
            self.export_button.config(state=tk.DISABLED)

        self.setup_treeview_columns_and_data()

    def setup_treeview_columns_and_data(self):
        self.clear_treeview()

        if self.current_results_df is None or self.current_results_df.empty:
            self.export_button.config(state=tk.DISABLED)
            self.tree["columns"] = ["Status"]
            self.tree.heading("Status", text="Status")
            self.tree.column("Status", anchor="center", width=300)
            status_message = "Load data and process to see results."
            if self.df_full is not None:
                status_message = "No valid data to display, or no objectives selected for score."
            self.tree.insert("", "end", values=(status_message,))
            return

        # Define the order of columns for display
        ordered_cols = list(self.input_identifier_cols)  # Start with identifier columns
        original_objective_cols = [oc['name'] for oc in self.objective_configs]
        ordered_cols.extend([col for col in original_objective_cols if col not in ordered_cols])
        normalized_cols_in_df = [oc['norm_col_name'] for oc in self.objective_configs if
                                 oc['norm_col_name'] in self.current_results_df.columns]
        ordered_cols.extend(normalized_cols_in_df)
        if 'Overall_Score' in self.current_results_df.columns:
            ordered_cols.append('Overall_Score')
        if 'Rank_within_Alloy' in self.current_results_df.columns:
            ordered_cols.append('Rank_within_Alloy')

        # Add any other columns from the original dataframe that weren't explicitly ordered
        for col in self.current_results_df.columns:
            if col not in ordered_cols:
                ordered_cols.append(col)

        self.current_results_df = self.current_results_df[ordered_cols]  # Reorder DataFrame columns
        cols_for_treeview = ordered_cols
        self.tree["columns"] = cols_for_treeview
        self.tree["show"] = "headings"

        header_font = tkFont.Font()  # Default font for measurement
        for col in cols_for_treeview:
            self.tree.heading(col, text=col, command=lambda c=col: self.sort_treeview_column(c))
            col_text_width = header_font.measure(col) + 30  # Add some padding

            # Define minimum widths for columns based on their names/types
            current_min_width = 90  # Default min width for general columns
            if "_norm" in col or col == "Overall_Score":
                current_min_width = 110
            elif col == "Composition":
                current_min_width = 150  # Identifier
            elif col == "Number density":
                current_min_width = 190
            elif col == "Mean radius":
                current_min_width = 180
            elif col == "Time [s]":
                current_min_width = 100
            elif col == "Volume fraction":
                current_min_width = 200
            elif col == "Matrix composition Mo":
                current_min_width = 180
            elif col == "Matrix composition C":
                current_min_width = 180
            elif col == "Rank_within_Alloy":
                current_min_width = 120

            self.tree.column(col, anchor="center", width=max(current_min_width, col_text_width), stretch=tk.NO)

        self.populate_treeview(self.current_results_df)
        if not self.current_results_df.empty:
            self.export_button.config(state=tk.NORMAL)

    def populate_treeview(self, df_to_display):
        self.clear_treeview()
        df_display = df_to_display.copy()  # Work on a copy for formatting

        for col in df_display.columns:
            # Skip string formatting for identifier columns if they are already strings/objects
            if col in self.input_identifier_cols:
                if pd.api.types.is_string_dtype(df_display[col]) or pd.api.types.is_object_dtype(df_display[col]):
                    continue

            if pd.api.types.is_numeric_dtype(df_display[col]):
                if "_norm" in col or col == "Overall_Score":
                    df_display[col] = df_display[col].round(4)
                elif col == "Mean radius (Nb-V)C":
                    df_display[col] = df_display[col].apply(
                        lambda x: f"{x:.3e}" if pd.notna(x) and x != 0 else (
                            f"{x:.7f}" if pd.notna(x) else x))
                elif col == "Number density (Nb-V)C":
                    df_display[col] = df_display[col].apply(
                        lambda x: f"{x:.3e}" if pd.notna(x) and x != 0 and abs(x) > 1e-4 else (
                            f"{x:.4f}" if pd.notna(x) else x))  # Handles 0 and small numbers
                elif col == "Time [s]":
                    df_display[col] = df_display[col].round(3)
                elif col in ["Volume fraction (Nb-V)C", "Matrix composition Mo", "Matrix composition C"]:
                    df_display[col] = df_display[col].round(4)  # Or another appropriate rounding
                elif col == "Rank_within_Alloy":
                    df_display[col] = df_display[col].astype(int)  # Show rank as integer
                else:  # Default rounding for other numeric columns
                    df_display[col] = df_display[col].round(3)

        for _, row_df_display in df_display.iterrows():
            self.tree.insert("", "end", values=list(row_df_display.astype(str)))

    def clear_treeview(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if self.current_results_df is None or self.current_results_df.empty:
            self.export_button.config(state=tk.DISABLED)

    def sort_treeview_column(self, column_name):
        if self.current_results_df is None or self.current_results_df.empty:
            return

        ascending_order = not self.last_sort_order_asc if column_name == self.last_sorted_column else True

        try:
            if column_name in self.current_results_df:
                sort_key_series = self.current_results_df[column_name]
                try:  # Attempt numeric sort first
                    numeric_key_series = pd.to_numeric(sort_key_series, errors='coerce')
                    if not numeric_key_series.isna().all():  # If at least one value is numeric
                        temp_df_for_sort = pd.DataFrame(
                            {'key': numeric_key_series, 'original_index': self.current_results_df.index})
                        temp_df_for_sort = temp_df_for_sort.sort_values(by='key', ascending=ascending_order,
                                                                        na_position='last')
                        sorted_df = self.current_results_df.loc[temp_df_for_sort['original_index']]
                    else:  # All coerced values are NaN or original was not numeric-like
                        raise ValueError("Column not numeric-like for sorting")
                except (ValueError, TypeError):  # Fallback to string sort if not purely numeric or coercion fails
                    # This handles identifier columns like 'Composition' or any other non-numeric data
                    sorted_df = self.current_results_df.sort_values(by=column_name, ascending=ascending_order,
                                                                    na_position='last',
                                                                    key=lambda col_series: col_series.astype(
                                                                        str).str.lower() if col_series.dtype == 'object' else col_series)
                self.current_results_df = sorted_df
        except Exception as e:
            print(f"Sorting error for column {column_name}: {e}. Falling back to default pandas sort.")
            self.current_results_df = self.current_results_df.sort_values(
                by=column_name, ascending=ascending_order, na_position='last'
            )

        self.last_sorted_column, self.last_sort_order_asc = column_name, ascending_order
        self.populate_treeview(self.current_results_df)

    def export_to_excel(self):
        if self.current_results_df is None or self.current_results_df.empty:
            messagebox.showwarning("No Data", "There is no data to export.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Results to Excel", defaultextension=".xlsx",
            filetypes=(("Excel files", "*.xlsx"), ("All files", "*.*"))
        )
        if not file_path:
            return

        try:
            # Export the current_results_df which contains all processed data
            # The dtypes in current_results_df should be correct for export
            df_to_export = self.current_results_df.copy()
            df_to_export.to_excel(file_path, index=False, engine='openpyxl')
            messagebox.showinfo("Export Successful", f"Data successfully exported to:\n{file_path}")
        except ImportError:
            messagebox.showerror("Export Error",
                                 "The 'openpyxl' library is required for Excel export.\n"
                                 "Please install it (e.g., 'pip install openpyxl') and try again.")
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred while exporting:\n{e}")


if __name__ == '__main__':
    main_root = tk.Tk()
    app = SelectiveDataProcessorApp(main_root)
    main_root.mainloop()