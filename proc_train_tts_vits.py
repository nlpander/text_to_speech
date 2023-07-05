import os
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from trainer import Trainer, TrainerArgs
import sys

from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from pydub import AudioSegment
import time


def main(argv):
    
    t0 = time.time()
    
    dataset_directory = os.getcwd()
    path = os.path.join(dataset_directory, 'ray_dalio.csv')
    df = pd.read_csv(path,index_col=None)

    df = df[df.text != 'None'].reset_index(drop=True)
    df = df.rename({'file':'audio_file'},axis=1)
    df['audio_file'] = df.audio_file.apply(lambda x:x.replace('2023Jun10_220252/', '').replace('.flac','.wav'))
    df['audio_file'] = df.audio_file.apply(lambda x:os.path.join(os.getcwd(),'wavs',x))
    df.to_csv('ray_dalio_.csv', sep='|')
    
    # LOAD DATA SAMPLES
    # Each sample is a list of ```[text, audio_file_path, speaker_name]```
    # You can define your custom sample loader returning the list of samples.
    # Or define your custom formatter and pass it to the `load_tts_samples`.
    # Check `TTS.tts.datasets.load_tts_samples` for more details.    
    
    samples = []

    for i in range(0,len(df)):
        tmp = {}
        tmp['text'] = df['text'].loc[i]
        tmp2 = df['audio_file'].loc[i]

        tmp['audio_file'] = tmp2
        tmp['speaker_name'] = 'ljspeech'
        tmp['language'] = 'en'
        tmp['audio_unique_name'] = f'ray_dalio_{tmp2}'
        samples.append(tmp)

    samples = list(np.random.permutation(samples))

    split = 0.9
    idx = int(split*len(samples))

    train_samples = samples[0:idx]
    eval_samples = samples[idx:]    
    
    test_sentences = ['The Bank Of England (BoE) appears to agree with you Warren in its latest report which states - the risks associated with Brexit have increased since June last year when the referendum took place.',\
    'Investors are also concerned about how sterling will react once Article 50 is triggered by Theresa May so there is no doubt that the pound will come under pressure at least initially after triggering Article 50.'\
                  'It would appear however that investor concerns regarding the outlook for GBP/USD are somewhat misplaced given that the market has already priced in a significant depreciation following an initial drop of around 13% against both EUR and USD on news of the vote result back in June.',
                  'The battle which posed a greater challenge was the one that took place within me.',
                  'I had to fight against my own desires and emotions, this is what made it so hard because these things were not external but rather they came from inside of me.']

    directory = os.getcwd()

    dataset_config = BaseDatasetConfig(
        dataset_name='ray_dalio_vits',
        formatter="ljspeech", 
        meta_file_train="ray_dalio_.csv", 
        path=directory,
        language='en-us'
    )

    audio_config = VitsAudioConfig(
        sample_rate=16000, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
    )

    num_epochs = 100
    batch_size = 1
    eval_batch_size = 1
    batch_group_size = 1

    config = VitsConfig(
        audio=audio_config,
        run_name="vits_ray_dalio",
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        batch_group_size=batch_group_size,
        num_loader_workers=2,
        num_eval_loader_workers=2,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=num_epochs,
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="en",
        phoneme_cache_path=os.path.join(directory, "phoneme_cache"),
        compute_input_seq_cache=True,
        print_step=25,
        print_eval=True,
        mixed_precision=False,
        output_path=directory,
        datasets=[dataset_config],
        cudnn_benchmark=True,
        test_sentences=test_sentences,
    )

    # INITIALIZE THE AUDIO PROCESSOR
    # Audio processor is used for feature extraction and audio I/O.
    # It mainly serves to the dataloader and the training loggers.
    ap = AudioProcessor.init_from_config(config)

    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # config is updated with the default characters if not defined in the config.
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # init model
    model = Vits(config, ap, tokenizer, speaker_manager=None)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(),
        config,
        os.path.join(os.getcwd(),'wavs'),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()
    print("Trained !!")
    
    print(f'Total training time elapsed .... {time.time() - t0}')


if __name__ == 'main':

    main(sys.argv)
    
