import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
import pickle


class EcephysAnalyzer:

    def __init__(self, structure_list=None, input_dir='data', output_dir='outputs',
                 spike_train_start_offset=0, spike_train_end=0.5):
        self.unit_ids_and_areas = None
        self.canrun = False
        self.session_to_analyze = None
        self.session_data = None
        self.presentations = None
        self.presentations_count = None
        self.drifting_gratings_spike_times = None
        self.drifting_gratings_spike_counts = None
        self.region_counts = None
        self.pivoted_df = None
        self.output_dir = os.path.join(output_dir, 'metadata')
        self.input_dir = input_dir
        self.manifest_path = os.path.join(self.input_dir, 'manifest.json')
        self.cache = EcephysProjectCache.from_warehouse(manifest=self.manifest_path)
        self.sessions = self.cache.get_session_table() # each session is a separate mouse experiment
        if structure_list is None:
            self.structure_list = ['VISp', 'VISl', 'VISal', 'VISam', 'VISpm', 'VISrl', 'LGd']
        else:
            self.structure_list = structure_list
        self.spike_train_start_offset = spike_train_start_offset
        self.spike_train_end = spike_train_end
        self.dt = 0.001


    def collate_sessions(self):
        all_units_with_metrics = self.cache.get_unit_analysis_metrics_by_session_type('brain_observatory_1.1')

        sessions_filtered = self.sessions[
            (self.sessions['ecephys_structure_acronyms'].apply(lambda x: set(self.structure_list).issubset(set(x)))) &
            (self.sessions['session_type'] == 'brain_observatory_1.1')]

        filtered_units = all_units_with_metrics[
            all_units_with_metrics['ecephys_session_id'].isin(sessions_filtered.index) &
            all_units_with_metrics['ecephys_structure_acronym'].isin(self.structure_list)]

        summary = filtered_units.groupby(['ecephys_session_id', 'ecephys_structure_acronym']).size().reset_index(
            name='count')
        self.unit_ids_and_areas = (filtered_units['ecephys_structure_acronym']
                                   .reset_index().rename(columns={'ecephys_unit_id': 'unit_id'}))
        self.pivoted_df = pd.pivot_table(summary, index='ecephys_session_id', columns='ecephys_structure_acronym',
                                         values='count')
        self.canrun = True


    def get_best_session(self, session_to_analyze=None):
        if not self.canrun:
            raise Exception('Cannot run without collating sessions first')
        if session_to_analyze is None:
            self.session_to_analyze = self.pivoted_df.min(axis=1).idxmax()
        else:
            self.session_to_analyze = session_to_analyze
        print('Getting data for session: ', self.session_to_analyze)
        self.session_data = self.cache.get_session_data(self.session_to_analyze)
        self.presentations = (self.session_data.get_stimulus_table(['drifting_gratings'])
                              .drop(['contrast', 'phase', 'size', 'spatial_frequency'], axis=1))
        self.presentations = self.presentations[self.presentations['orientation'] != 'null'].sort_values('stimulus_condition_id')
        # SAME FUNCTIONALITY
        # presentations = session_data.stimulus_presentations
        # presentations = presentations[presentations['stimulus_name']. \
        #     isin(['drifting_gratings', 'spontaneous'])]
        # presentations = presentations.drop(presentations.columns[presentations.eq('null').all()], axis=1)
        presentations_count = self.presentations.groupby(['stimulus_name', 'stimulus_condition_id']). \
            size().reset_index(name='num_trials')
        conditions = self.session_data.stimulus_conditions
        conditions = conditions[conditions['stimulus_name'] == 'drifting_gratings']
        conditions = conditions.drop(conditions.columns[conditions.eq('null').all()], axis=1)
        conditions = conditions[conditions['orientation'] != 'null']
        conditions = conditions.drop(['contrast', 'mask', 'opacity', 'phase', 'size',
                                      'spatial_frequency', 'units', 'color_triplet',
                                      'stimulus_name'], axis=1)
        # join conditions to presentation_count by condition.index and presentation_count.stimulus_condition_id
        self.presentations_count = presentations_count.merge(conditions, left_on='stimulus_condition_id',
                                                             right_index=True)
        print('Getting spike times for session: ', self.session_to_analyze)
        for i, id in enumerate(self.presentations.index.values):
            if i == 0:
                drifting_gratings_spike_times = self.session_data.presentationwise_spike_times(stimulus_presentation_ids=id).reset_index()
            else:
                drifting_gratings_spike_times = pd.concat([drifting_gratings_spike_times, self.session_data.presentationwise_spike_times(stimulus_presentation_ids=id).reset_index()])
        stop = self.spike_train_end + self.dt
        # select only spikes that happen before stop
        drifting_gratings_spike_times = drifting_gratings_spike_times[drifting_gratings_spike_times['time_since_stimulus_presentation_onset'] < stop]
        drifting_gratings_spike_times_count = drifting_gratings_spike_times.groupby('unit_id').size().reset_index(name='unit_spike_count_across_trials')
        # select all unit-trials with more than 5 spikes over the course of the trial
        drifting_gratings_spike_times_count = drifting_gratings_spike_times_count[drifting_gratings_spike_times_count['unit_spike_count_across_trials'] > 0]
        # compute the minimum time since stimulus presentation onset for each unit_id, stimulus_presentation_id pair
        unit_time_of_first_spike = (drifting_gratings_spike_times.groupby('unit_id')
                                    .apply(lambda x: x['time_since_stimulus_presentation_onset'].min())
                                    .reset_index(name='unit_time_of_first_spike'))
        # select unit-trials where the first spike occurs within (stop/1.5) seconds of stimulus presentation onset
        unit_time_of_first_spike = unit_time_of_first_spike[unit_time_of_first_spike['unit_time_of_first_spike'] < (stop/1.1)]
        # select the rows in the spike_times table where both the unit_id and stimulus_presentation_id are in the spike_times_count table
        # drifting_gratings_spike_times restricts the session. unit_ids_and_areas restricts the regions. The other two tables restrict (exceptionally bad) units.
        self.drifting_gratings_spike_times = drifting_gratings_spike_times.merge(
            drifting_gratings_spike_times_count, on='unit_id').merge(
            unit_time_of_first_spike, on='unit_id').merge(
            self.presentations, on='stimulus_presentation_id').merge(
            self.unit_ids_and_areas, on='unit_id')
        self.unit_ids_and_areas = self.drifting_gratings_spike_times[['unit_id', 'ecephys_structure_acronym']].drop_duplicates()
        unit_id_map = {unit_id: idx+1 for idx, unit_id in enumerate(self.unit_ids_and_areas['unit_id'].unique())}
        self.unit_ids_and_areas['unit_id_int'] = self.unit_ids_and_areas['unit_id'].map(unit_id_map)
        self.region_counts = (self.drifting_gratings_spike_times.groupby('ecephys_structure_acronym')['unit_id']
                              .nunique().reset_index(name='num_units'))
        print('Getting spike counts for session: ', self.session_to_analyze)
        num = int((stop+self.spike_train_start_offset)/self.dt)
        time_bin_edges = np.linspace(-self.spike_train_start_offset, stop, num)  # 2050 edges, 2049 bins
        # Dimensions: (stimulus_presentation_id, time_relative_to_stimulus_onset, unit_id)
        # This will select all trials for each unit, so the fitering we did earlier was really to remove really bad units
        self.drifting_gratings_spike_counts = self.session_data.presentationwise_spike_counts(
            stimulus_presentation_ids=self.presentations.index.values,
            bin_edges=time_bin_edges,
            unit_ids=self.unit_ids_and_areas['unit_id'])


    def filter_spike_times(self, unit_ids, presentation_ids, regions, conditions, unit_as_index=True):
        if isinstance(unit_ids, int):
            unit_ids = [unit_ids]
        if isinstance(presentation_ids, int):
            presentation_ids = [presentation_ids]
        if isinstance(regions, str):
            regions = [regions]
        if isinstance(conditions, int):
            conditions = [conditions]
        data = self.drifting_gratings_spike_times
        if presentation_ids is not None:
            data = data[data['stimulus_presentation_id'].isin(presentation_ids)]
        elif conditions is not None:
            data = data[data['stimulus_condition_id'].isin(conditions)]
        if unit_ids is not None:
            if unit_as_index:
                unit_ids_for_index = self.unit_ids_and_areas[
                    self.unit_ids_and_areas['unit_id_int'].isin(unit_ids)]['unit_id'].values
            else:
                unit_ids_for_index = unit_ids
            data = data[data['unit_id'].isin(unit_ids_for_index)]
        elif regions is not None:
            data = data[data['ecephys_structure_acronym'].isin(regions)]
        return data

    def filter_spike_counts(self, unit_ids, presentation_ids, regions, conditions, unit_as_index=True):
        if isinstance(unit_ids, int):
            unit_ids = [unit_ids]
        if isinstance(presentation_ids, int):
            presentation_ids = [presentation_ids]
        if isinstance(regions, str):
            regions = [regions]
        if isinstance(conditions, int):
            conditions = [conditions]
        data = self.drifting_gratings_spike_counts
        if presentation_ids is not None:
            data = data.sel(stimulus_presentation_id=presentation_ids)
        elif conditions is not None:
            presentation_ids_for_condition = self.presentations[
                self.presentations['stimulus_condition_id'].isin(conditions)].index.values
            data = data.sel(stimulus_presentation_id=presentation_ids_for_condition)
        if unit_ids is not None:
            if unit_as_index:
                unit_ids_for_index = self.unit_ids_and_areas[
                    self.unit_ids_and_areas['unit_id_int'].isin(unit_ids)]['unit_id'].values
            else:
                unit_ids_for_index = unit_ids
            data = data.sel(unit_id=unit_ids_for_index)
        elif regions is not None:
            unit_ids_for_region = self.unit_ids_and_areas[
                self.unit_ids_and_areas['ecephys_structure_acronym'].isin(regions)]['unit_id'].values
            data = data.sel(unit_id=unit_ids_for_region)
        return data

    def initialize(self):
        self.collate_sessions()
        self.get_best_session()
        return self

    def plot_presentations_times(self, save_dir):

        save_dir = os.path.join(self.output_dir, save_dir)
        os.makedirs(save_dir, exist_ok=True)

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
        plt.savefig(os.path.join(save_dir, 'presentation_times.png'))
        plt.close()

    def plot_spike_times(self, save_dir, unit_ids=None, presentation_ids=None, regions=None, conditions=None):

        save_dir = os.path.join(self.output_dir, save_dir)
        os.makedirs(save_dir, exist_ok=True)

        data = self.filter_spike_times(conditions, presentation_ids, regions, unit_ids).sort_values([
            'stimulus_condition_id',
            'stimulus_presentation_id',
            'ecephys_structure_acronym',
            'unit_id',
            'time_since_stimulus_presentation_onset'])

        data.loc[:, 'y_axis'] = data[['stimulus_presentation_id', 'unit_id']].astype(str).agg('_'.join, axis=1)
        data.plot(x='time_since_stimulus_presentation_onset', y='y_axis', kind='scatter', s=1,
                  c=data['ecephys_structure_acronym'].astype('category').cat.codes, cmap='viridis',
                  yticks=[], figsize=(7, 8))
        plt.minorticks_on()
        plt.grid(axis='x', which='major', linestyle='--', linewidth='0.4', color='grey')
        plt.xlabel('Time since stimulus onset')
        plt.ylabel('Units grouped by conditions')
        plt.savefig(os.path.join(save_dir, 'spike_times_by_conditions.png'))
        plt.close()

        data = self.filter_spike_times(conditions, presentation_ids, regions, unit_ids).sort_values([
            'ecephys_structure_acronym',
            'stimulus_condition_id',
            'stimulus_presentation_id',
            'unit_id',
            'time_since_stimulus_presentation_onset'])

        data.loc[:, 'y_axis'] = data[['stimulus_presentation_id', 'unit_id']].astype(str).agg('_'.join, axis=1)
        data.plot(x='time_since_stimulus_presentation_onset', y='y_axis', kind='scatter', s=1,
                  c=data['stimulus_condition_id'].astype('category').cat.codes, cmap='viridis',
                  yticks=[], figsize=(7, 8))
        plt.minorticks_on()
        plt.grid(axis='x', which='major', linestyle='--', linewidth='0.4', color='grey')
        plt.xlabel('Time since stimulus onset')
        plt.ylabel('Units grouped by regions')
        plt.savefig(os.path.join(save_dir, 'spike_times_by_regions.png'))
        plt.close()

    def plot_spike_counts(self, save_dir, unit_ids=None, presentation_ids=None, regions=None, conditions=None):

        save_dir = os.path.join(self.output_dir, save_dir)
        os.makedirs(save_dir, exist_ok=True)

        data = self.filter_spike_counts(unit_ids, presentation_ids, regions, conditions)

        # sum over the stimulus_presentation_id dimension anc convert to dataframe
        data = (data.sum(dim='stimulus_presentation_id').to_dataframe(name='spike_counts')
                .pivot_table(index='time_relative_to_stimulus_onset', columns='unit_id', values='spike_counts'))
        data = data.loc[:, data.sum().sort_values(ascending=False).index.values]

        # Iterate over sets of {per_plot} columns (units) and save to output folder
        c = data.shape[1]
        per_plot = 9
        for i in range(0, c, per_plot):
            # Select the next set of 10 columns
            a_subset = data.iloc[:, i:i+per_plot]
            # Create a figure with 10 subplots
            fig, axs = plt.subplots(nrows=per_plot, sharex=True, figsize=(7, 12))
            # Plot each column of a_subset on a separate subplot
            for j, col in enumerate(a_subset.columns):
                axs[j].plot(a_subset.index, a_subset[col])
                axs[j].set_ylabel(self.unit_ids_and_areas[self.unit_ids_and_areas["unit_id"]==col]["unit_id_int"].values[0])
                axs[j].yaxis.set_label_position("right")
            # Add labels to the plot
            fig.suptitle(f'Columns {i + 1}-{i + per_plot} of {c}')
            plt.xlabel('Time relative to stimulus onset')
            # Save the plot to output_dir
            fig.savefig(os.path.join(save_dir, f'units_{i + 1}-{i + per_plot}.png'))
            plt.close(fig)


    def sample_data(self, unit_ids=None, presentation_ids=None, regions=None, conditions=None, num_factors=3):

        spike_time_info = self.filter_spike_times(unit_ids, presentation_ids, regions, conditions) # these have only those units that fired within 0.5 seconds of stimulus onset, and for those, it keeps the trials where the unit has more than 10 spikes over the course of the trial.
        relevant_conditions = spike_time_info['stimulus_condition_id'].unique()
        C = relevant_conditions.shape[0]
        A = spike_time_info['ecephys_structure_acronym'].unique().shape[0]
        times_data_units = spike_time_info['unit_id'].unique()
        count_data = self.filter_spike_counts(times_data_units, presentation_ids, regions, relevant_conditions, unit_as_index=False)
        time = count_data['time_relative_to_stimulus_onset'].values
        # Y # K x T x R x C
        Y = count_data.to_numpy().T.astype(int)
        Y = Y.reshape(*Y.shape[:2], -1, C)
        # neuron_factor_access # K x L x C
        neuron_factor_access = np.zeros((Y.shape[0], num_factors*A, C))
        unit_areas = self.unit_ids_and_areas[self.unit_ids_and_areas['unit_id'].isin(count_data['unit_id'].values)]['ecephys_structure_acronym'].values
        unique_regions = np.unique(unit_areas)
        for idx, region in enumerate(unique_regions):
            area_start_indx = idx * num_factors
            neuron_factor_access[np.where(unit_areas==region), area_start_indx:(area_start_indx + num_factors), :] = 1
        # Group neurons by region
        sorted_indices = (pd.DataFrame(np.concatenate(neuron_factor_access, axis=-1).T).
                          sort_values(by=list(np.arange(neuron_factor_access.shape[1])), ascending=False).index)
        neuron_factor_access = np.concatenate(neuron_factor_access, axis=-1).T[sorted_indices].reshape(Y.shape[0], C, num_factors*A).transpose(0, 2, 1)
        Y = np.concatenate(Y, axis=-1).T[sorted_indices].reshape(Y.shape[0], C, Y.shape[2], Y.shape[1]).transpose(0, 3, 2, 1)
        return Y, time, neuron_factor_access, spike_time_info


    def save_sample(self, Y, time, neuron_factor_access, spike_time_info, folder_name):
        save_dir = os.path.join(self.output_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        save_dir = os.path.join(save_dir, f'{folder_name}.pkl')
        print('Saving sample to: ', save_dir)
        with open(save_dir, 'wb') as f:
            pickle.dump({'Y': Y, 'time': time, 'neuron_factor_access': neuron_factor_access, 'spike_time_info': spike_time_info}, f)


    def load_sample(self, folder_name):
        save_dir = os.path.join(self.output_dir, folder_name, f'{folder_name}.pkl')
        if not os.path.exists(save_dir):
            print('File not found: ', save_dir)
            return None, None, None, None
        print('Loading sample from: ', save_dir)
        with open(save_dir, 'rb') as f:
            data = pickle.load(f)
        return data['Y'], data['time'], data['neuron_factor_access'], data['spike_time_info']