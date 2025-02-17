import os, re, pickle
import pandas as pd
import numpy as np
import os, torch, glob
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from utils.timefeatures import time_features
from utils.sktime import load_from_tsfile_to_dataframe
from data.uea import subsample, interpolate_missing, Normalizer
import warnings

warnings.filterwarnings('ignore')

def add_time_features(dates, timeenc=0, freq='h'):
    df_stamp = pd.DataFrame()
    df_stamp['date'] = pd.to_datetime(dates)
    if timeenc == 0:
        df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
        df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
        df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
        df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
        data_stamp = df_stamp.drop(['date'], 1).values
    elif timeenc == 1:
        data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=freq)
        data_stamp = data_stamp.transpose(1, 0)
    return data_stamp

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='electricity.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.8)
        num_test = int(len(df_raw) * 0.1)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # choose input data based on Multivariate or Univariate setting
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # scale data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # add time encoding
        df_stamp = df_raw[['date']][border1:border2]
        
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # select data split
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data
    

class Dataset_Pred(Dataset):
    def __init__(
        self, root_path, flag='pred', size=None,
        features='S', data_path='ETTh1.csv',
        target='OT', scale=True, inverse=False, 
        timeenc=0, freq='h', cols=None
        ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data


class MultiTimeSeries(Dataset):
    def __init__(
        self, root_path, flag='train', size=None,
        features='S', data_path='ETTh1.csv',
        target='OT', scale=True, timeenc=0, freq='d', 
        time_col='Date', id_col='FIPS', max_samples=-1
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        
        self.id_col = id_col
        self.time_col = time_col
        self.time_steps = self.seq_len + self.pred_len
        self.max_samples = max_samples
        self.scaler = StandardScaler()
        self.__read_data__()
        
    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path,self.data_path))
        df_raw[self.time_col] = pd.to_datetime(df_raw[self.time_col])
        
        id_col, time_col, target, time_steps = self.id_col, self.time_col, self.target, self.time_steps
        df_raw.sort_values(by=time_col, inplace=True)
        input_cols = [
            col for col in df_raw.columns \
                if col not in [id_col, time_col, target]
        ]
            
        dates = df_raw[time_col].unique()
        num_total = len(dates)
        num_test = self.pred_len # int(len(dates) * 0.2)
        num_vali = self.pred_len # num_total - num_train - num_test
        num_train = num_total - num_test -  num_vali# int(len(dates) * 0.7)
        
        border1s = [0, num_train - self.seq_len, num_total - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, num_total]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]
        
        border1 = dates[border1s[self.set_type]]
        border2 = dates[border2s[self.set_type]-1]
        border1 = df_raw[time_col].values.searchsorted(border1, side='left')
        border2 = df_raw[time_col].values.searchsorted(border2, side='right')
        
        # get input features
        if self.features == 'M' or self.features == 'MS':
            selected_columns = input_cols+[target]
        elif self.features == 'S':
            selected_columns = [target]
        print('Selected columns ', selected_columns)
        self.selected_columns = selected_columns
            
        df_data = df_raw[border1:border2].copy().reset_index(drop=True)
        
        if self.scale:
            train_end = df_raw[time_col].values.searchsorted(
                dates[border2s[0]-1], side='right'
            )
            train_data = df_raw[0:train_end]
            self.scaler.fit(train_data[selected_columns])
            df_data.loc[:, selected_columns] = self.scaler.transform(df_data[selected_columns])
            
        # add time encoding
        data_stamp = self._add_time_features(df_data.loc[0, [self.time_col]])
        time_encoded_columns = data_stamp.shape[1]
        print('Number of time encoded columns :', time_encoded_columns)
        
        print('Getting valid sampling locations.')
        
        valid_sampling_locations = []
        split_data_map = {}
        for identifier, df in df_data.groupby(id_col):
            num_entries = len(df)
            if num_entries >= time_steps:
                valid_sampling_locations += [
                    (identifier, i)
                    for i in range(num_entries - time_steps + 1)
                ]
            split_data_map[identifier] = df

        max_samples = self.max_samples # -1 takes all samples
        
        if max_samples > 0 and len(valid_sampling_locations) > max_samples:
            print('Extracting {} samples...'.format(max_samples))
            ranges = [valid_sampling_locations[i] for i in np.random.choice(
                  len(valid_sampling_locations), max_samples, replace=False)]
        else:
            # print('Max samples={} exceeds # available segments={}'.format(
            #      max_samples, len(valid_sampling_locations)))
            ranges = valid_sampling_locations
            max_samples = len(valid_sampling_locations)
        
        self.data = np.zeros((max_samples, self.time_steps, len(selected_columns)))
        self.data_stamp = np.zeros((max_samples, self.time_steps, time_encoded_columns))
        for i, tup in enumerate(ranges):
            if ((i + 1) % 10000) == 0:
                print(i + 1, 'of', max_samples, 'samples done...')
            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx:start_idx + time_steps]
            self.data[i] = sliced[selected_columns]
            self.data_stamp[i] = add_time_features(
                sliced[[self.time_col]], self.timeenc, self.freq
            )
        
    def __getitem__(self, index):
        s_end = self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data[index][:s_end]
        seq_y = self.data[index][r_begin:r_end]
        
        seq_x_mark = self.data_stamp[index][:s_end]
        seq_y_mark = self.data_stamp[index][r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data) # - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        if self.scale:
            return self.scaler.inverse_transform(data)
        return data
    
    
class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, root_path, file_list=None, limit_size=None, flag=None):
        self.root_path = root_path
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
               torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values)

    def __len__(self):
        return len(self.all_IDs)
    
class MimicIII(Dataset):
    def __init__(
        self, root_path='./dataset/mimic_iii', flag='train', 
        data_path='patient_vital_preprocessed.pkl', 
        scale=True, size= [0.8, 0.1, 0.1], seed=7, seq_len=48
    ):
        # init
        assert flag in ['train', 'test', 'val']
        self.flag = flag
        
        vital_IDs = ['HeartRate' , 'SysBP' , 'DiasBP' , 'MeanBP' , 'RespRate' , 'SpO2' , 'Glucose' ,'Temp']
        others = ['gender', 'age', 'ethnicity', 'first_icu_stay']
        lab_IDs = [
            'ANION GAP', 'ALBUMIN', 'BICARBONATE', 'BILIRUBIN', 
            'CREATININE', 'CHLORIDE', 'GLUCOSE', 'HEMATOCRIT', 
            # 'HEMOGLOBIN' 'LACTATE' -> 'HEMOGLOBIN', 'LACTATE'. But the source preprocessing uses it like this
            'HEMOGLOBIN' 'LACTATE', 'MAGNESIUM', 'PHOSPHATE', 
            'PLATELET', 'POTASSIUM', 'PTT', 'INR', 
            'PT', 'SODIUM', 'BUN', 'WBC'
        ]
        self.feature_columns = vital_IDs + others + lab_IDs

        self.scale = scale
        self.size = size
        self.seed = seed
        
        self.num_classes = 2
        self.class_names = [0, 1]
        self.seq_len = seq_len

        self.root_path = root_path
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.__read_data__()

    def __read_data__(self):
        filepath = os.path.join(self.root_path, self.data_path)
        with open(filepath, 'rb') as input_file:
            data = pickle.load(input_file)

        X = np.array([row[0] for row in data])
        # batch x features x seq_len -> batch x seq_len x features
        X = X.transpose((0, 2, 1))
        assert self.seq_len <= X.shape[1], f'Please set seq_len smaller than {X.shape[1]}'
        # save the most recent sequence
        X = X[:, -self.seq_len:]
        
        Y = np.array([int(row[1]) for row in data])
        
        # split sizes
        n_total, self.max_seq_len, self.n_features = X.shape
        num_vali = round(n_total * self.size[1])
        num_test= round(n_total * self.size[2])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, random_state=self.seed, shuffle=True, 
            test_size=num_test, stratify=Y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, random_state=self.seed, shuffle=True, 
            test_size=num_vali, stratify=y_train
        )
        
        if self.flag == 'train': X, Y = X_train, y_train
        elif self.flag == 'val': X, Y = X_val, y_val
        else: X, Y = X_test, y_test
        print(f'Dead patients {sum(Y)}, percentage {100*sum(Y)/len(Y):.2f}.')
            
        labels = pd.Series(Y, dtype='category')
        self.class_names = labels.cat.categories
        self.Y = np.array(labels.cat.codes).reshape(-1, 1)
        
        if self.scale:
            self.scaler.fit(X_train.reshape((X_train.shape[0], -1)))
            
            original_shape = X.shape
            X = self.scaler.transform(
                X.reshape((X.shape[0], -1))
            ).reshape(original_shape)
        
        self.X = X

    def __getitem__(self, index):
        X = torch.from_numpy(self.X[index])
        Y = torch.Tensor(self.Y[index])
        return X, Y
            
    
    def __len__(self):
        return self.X.shape[0]

    def inverse_transform(self, data):
        if self.scale:
            original_shape = data.shape
            return self.scaler.inverse_transform(
                data.reshape((data.shape[0], -1))
            ).reshape(original_shape)
            
        return data