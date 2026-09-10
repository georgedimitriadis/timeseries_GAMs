from tbparse import SummaryReader
import pandas as pd
import numpy as np

def get_dfs_from_events(tensorboard_file: str):
    reader = SummaryReader(tensorboard_file)
    df = reader.scalars

    total_steps = df['step'].max() + 1
    total_epochs = df.loc[total_steps - 1, 'value'] + 1
    steps_per_epoch = total_steps // total_epochs

    df_train_loss = pd.DataFrame(columns=['epoch', 'epoch_step', 'train_loss'], index=np.arange(total_steps), dtype=float)
    df_train_loss['train_loss'] = np.array(df['value'][df['tag']=='train_loss'])
    for k in range(len(df_train_loss)):
        df_train_loss.loc[k, 'epoch'] = int(k // steps_per_epoch)
        df_train_loss.loc[k, 'epoch_step'] = int(k % steps_per_epoch)

    df_validation = pd.DataFrame(columns=['CRPS', 'NMSE'], index=range(int(total_epochs)))
    df_validation['CRPS'] = np.array(df['value'][df['tag']=='val_CRPS'])
    df_validation['NMSE'] = np.array(df['value'][df['tag']=='val_ND'])

    return df_train_loss, df_validation

