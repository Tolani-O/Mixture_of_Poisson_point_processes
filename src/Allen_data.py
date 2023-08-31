import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache


class EcephysAnalyzer:

    def __init__(self, input_dir='data', output_dir='outputs'):
        self.unit_ids_and_areas = None
        self.canrun = False
        self.session_to_analyze = None
        self.session_data = None
        self.presentations = None
        self.presentations_count = None
        self.drifting_gratings_spike_times = None
        self.region_counts = None
        self.pivoted_df = None
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.manifest_path = os.path.join(self.input_dir, 'manifest.json')
        self.cache = EcephysProjectCache.from_warehouse(manifest=self.manifest_path)
        self.sessions = self.cache.get_session_table()
        self.structure_list = ['VISp', 'VISl', 'VISal']

    def collate_sessions(self):
        all_units_with_metrics = self.cache.get_unit_analysis_metrics_by_session_type(
            'brain_observatory_1.1',
            amplitude_cutoff_maximum=np.inf,
            presence_ratio_minimum=-np.inf,
            isi_violations_maximum=np.inf)

        sessions_filtered = self.sessions[
            (self.sessions['ecephys_structure_acronyms'].apply(lambda x: set(self.structure_list).issubset(set(x)))) &
            (self.sessions['session_type'] == 'brain_observatory_1.1')]

        filtered_units = all_units_with_metrics[
            (all_units_with_metrics['isi_violations'] < 0.5) &
            (all_units_with_metrics['amplitude_cutoff'] < 0.1) &
            all_units_with_metrics['ecephys_session_id'].isin(sessions_filtered.index) &
            all_units_with_metrics['ecephys_structure_acronym'].isin(self.structure_list)]

        summary = filtered_units.groupby(['ecephys_session_id', 'ecephys_structure_acronym']).size().reset_index(
            name='count')
        self.unit_ids_and_areas = (filtered_units['ecephys_structure_acronym']
                                   .reset_index().rename(columns={'ecephys_unit_id': 'unit_id'}))
        self.pivoted_df = pd.pivot_table(summary, index='ecephys_session_id', columns='ecephys_structure_acronym',
                                         values='count')
        self.canrun = True

        return self.pivoted_df

    def get_best_session(self):
        if not self.canrun:
            raise Exception('Cannot run without collating sessions first')
        self.session_to_analyze = self.pivoted_df['VISl'].idxmax()
        self.session_data = self.cache.get_session_data(self.session_to_analyze,
                                                        isi_violations_maximum=0.5,
                                                        amplitude_cutoff_maximum=0.1,
                                                        presence_ratio_minimum=-np.inf)
        self.presentations = (self.session_data.get_stimulus_table(['drifting_gratings'])
                              .drop(['contrast', 'phase', 'size', 'spatial_frequency'], axis=1))
        # SAME FUNCTIONALITY
        # presentations = session_data.stimulus_presentations
        # presentations = presentations[presentations['stimulus_name']. \
        #     isin(['drifting_gratings', 'spontaneous'])]
        # presentations = presentations.drop(presentations.columns[presentations.eq('null').all()], axis=1)
        presentations_count = self.presentations.groupby(['stimulus_name', 'stimulus_condition_id']). \
            size().reset_index(name='num_trials')
        conditions = self.session_data.stimulus_conditions
        conditions = conditions[conditions['stimulus_name'].isin(['drifting_gratings'])]
        conditions = conditions.drop(conditions.columns[conditions.eq('null').all()], axis=1)
        conditions = conditions.drop(['contrast', 'mask', 'opacity', 'phase', 'size',
                                      'spatial_frequency', 'units', 'color_triplet',
                                      'stimulus_name'], axis=1)
        # join conditions to presentation_count by condition.index and presentation_count.stimulus_condition_id
        self.presentations_count = presentations_count.merge(conditions, left_on='stimulus_condition_id',
                                                             right_index=True)
        drifting_gratings_spike_times = self.session_data.presentationwise_spike_times(
            stimulus_presentation_ids=self.presentations.index.values).reset_index()
        drifting_gratings_spike_times_count = (
            drifting_gratings_spike_times.groupby(['unit_id', 'stimulus_presentation_id'])
            .size().reset_index(name='unit_spike_count_on_trial'))
        # select all units with more than 10 spikes over the course of the trial
        drifting_gratings_spike_times_count = drifting_gratings_spike_times_count[
            drifting_gratings_spike_times_count['unit_spike_count_on_trial'] > 10]
        # select the minimum time since stimulus presentation onset for each unit_id, stimulus_presentation_id pair
        unit_time_of_first_spike = (drifting_gratings_spike_times.groupby(['unit_id', 'stimulus_presentation_id'])
                                    .apply(lambda x: x['time_since_stimulus_presentation_onset'].min())
                                    .reset_index(name='unit_time_of_first_spike'))
        # select units that started firing within 0.5 seconds of stimulus presentation onset
        unit_time_of_first_spike = unit_time_of_first_spike[unit_time_of_first_spike['unit_time_of_first_spike'] < 0.5]
        # select the rows in the spike_times table where both the unit_id and stimulus_presentation_id are in the spike_times_count table
        self.drifting_gratings_spike_times = drifting_gratings_spike_times.merge(
            drifting_gratings_spike_times_count, on=['unit_id', 'stimulus_presentation_id']).merge(
            unit_time_of_first_spike, on=['unit_id', 'stimulus_presentation_id']).merge(
            self.presentations, on='stimulus_presentation_id').merge(
            self.unit_ids_and_areas, on='unit_id')
        self.unit_ids_and_areas = self.drifting_gratings_spike_times[['unit_id', 'ecephys_structure_acronym']].drop_duplicates()
        self.region_counts = (self.drifting_gratings_spike_times.groupby('ecephys_structure_acronym')['unit_id']
                              .nunique().reset_index(name='num_units'))
        time_bin_edges = np.linspace(-0.05, 2, 2050)  # 2050 edges, 2049 bins
        # Dimensions: (stimulus_presentation_id, time_bin, unit_id)
        self.drifting_gratings_spike_counts = self.session_data.presentationwise_spike_counts(
            stimulus_presentation_ids=self.presentations.index.values,
            bin_edges=time_bin_edges,
            unit_ids=self.unit_ids_and_areas['unit_id'])
        return self.session_to_analyze

    def plot_presentations_times(self):
        x_err = [self.presentations['duration'] / 2, self.presentations['duration'] / 2]
        plt.errorbar(self.presentations['start_time'] + x_err[0],
                     self.presentations['stimulus_name'],
                     xerr=x_err, ecolor='black', linestyle='')
        # Add labels to the plot
        plt.xlabel('Start Time')
        # remove the y axis text
        plt.yticks([])
        plt.ylabel('Drifting Gratings')
        # Show the plot
        plt.show()

    def plot_spike_times(self, region=None, condition=None, unit_id=None):
        data = self.drifting_gratings_spike_times
        if unit_id is not None:
            data = data[data['unit_id'] == unit_id]
        elif region is not None:
            data = data[data['ecephys_structure_acronym'] == region]
        if condition is not None:
            data = data[data['stimulus_condition_id'] == condition]
        data['stimulus_presentation_id'] = data['stimulus_presentation_id'].astype(str)
        data.plot(x='time_since_stimulus_presentation_onset', y='stimulus_presentation_id', kind='scatter', s=1, yticks=[])
        plt.title(f'{region} {condition} {unit_id}')
        plt.show()

    def plot_spike_counts(self, region=None, condition=None, unit_id=None, stop_time=0.5):
        data = self.drifting_gratings_spike_counts
        if condition is not None:
            presentation_ids_for_condition = self.presentations[
                self.presentations['stimulus_condition_id'] == condition].index.values
            data = data.sel(stimulus_presentation_id=presentation_ids_for_condition)
        if unit_id is not None:
            data = data.sel(unit_id=unit_id)
        elif region is not None:
            unit_ids_for_region = self.unit_ids_and_areas[
                self.unit_ids_and_areas['ecephys_structure_acronym'] == region]['unit_id'].values
            data = data.sel(unit_id=unit_ids_for_region)
        # sum over the stimulus_presentation_id dimension anc convert to dataframe
        data = (data.sum(dim='stimulus_presentation_id').to_dataframe(name='spike_counts')
                .pivot_table(index='time_relative_to_stimulus_onset', columns='unit_id', values='spike_counts'))
        # Select only units for which the maximum value is greater than 2
        stop_time = (stop_time * 1000) + 50
        data = data.loc[:stop_time, data.max() > 2]

        # Iterate over sets of 10 columns (units) and save to output folder
        c = data.shape[1]
        per_plot = 10
        for i in range(0, c, per_plot):
            # Select the next set of 10 columns
            a_subset = data.iloc[:stop_time, i:i+per_plot]
            # Create a figure with 10 subplots
            fig, axs = plt.subplots(nrows=per_plot, sharex=True)
            # Plot each column of a_subset on a separate subplot
            for j, col in enumerate(a_subset.columns):
                axs[j].plot(a_subset.index, a_subset[col])
            # Add labels to the plot
            fig.suptitle(f'Columns {i + 1}-{i + per_plot} of {c}')
            plt.xlabel('Time relative to stimulus onset')
            plt.ylabel('PSTH')
            # Save the plot to output_dir
            fig.savefig(os.path.join(self.output_dir, f'columns_{i + 1}-{i + per_plot}.png'))

    def sample_data(self):
        data = self.drifting_gratings_spike_counts
        unit_ids_for_region = self.unit_ids_and_areas[
            self.unit_ids_and_areas['ecephys_structure_acronym'] == 'VISp']['unit_id'].values
        data = (data
                .sel(unit_id=unit_ids_for_region)
                .sel(stimulus_presentation_id=49417)
                .to_dataframe(name='spike_counts')
                .pivot_table(index='time_relative_to_stimulus_onset',
                             columns='unit_id', values='spike_counts'))
        return data.iloc[:550, :].to_numpy().T

    def run(self):
        self.collate_sessions()
        self.get_best_session()
        return self.sample_data()
