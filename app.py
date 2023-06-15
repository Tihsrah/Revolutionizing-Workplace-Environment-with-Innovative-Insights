import subprocess
# # Run the pip install command
# subprocess.check_call(['pip', 'install', 'wordcloud'])
# subprocess.check_call(['pip', 'install', 'git+https://github.com/openai/whisper.git'])
# subprocess.check_call(['pip', 'install', 'transformers'])
# subprocess.check_call(['pip', 'install', 'imageio==2.4.1'])
# subprocess.check_call(['pip', 'install', 'moviepy'])
# subprocess.check_call(['pip', 'install', 'keybert'])
# subprocess.check_call(['pip', 'install', 'pytube'])
import evaluate
import datasets
import streamlit as st
import os
from wordcloud import WordCloud
from keybert import KeyBERT
import pandas as pd
import matplotlib.pyplot as plt
# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
from transformers import AutoTokenizer

from moviepy.editor import *
from tqdm import tqdm
import os
import math
import nltk
nltk.download('punkt')
import whisper
from transformers import pipeline
import joblib
from pytube import YouTube
clip_run_range=0
clip_duration=0
def process_video(path):
    # whisper_model = whisper.load_model("base")
    # joblib.dump(whisper_model, 'whisper_model.joblib')
    whisper_model = joblib.load('whisper_model.joblib')
    
    def SpeechToTextEng(aud_path):
      print("auth path",aud_path)
      result = whisper_model.transcribe(aud_path)
      return result["text"]
        
    def run_range(duration):
        time=duration/60
        floor=math.ceil(time)
        return floor

    time_range=60
    

    def audio_generator(path,aud=0,vid=0):
        global clip_run_range
        global clip_duration
        if vid==1:
            clip=VideoFileClip(path)
            clip_duration = clip.duration
            clip_run_range=run_range(clip_duration)
            for i in range(clip_run_range):
                left=i*time_range
                right=left+time_range
                # print(left,right)

                crop_clip=clip.subclip(left,right)
                try:
                    crop_clip.audio.write_audiofile("vid_to_aud"+str(i)+".mp3")
                except:
                    pass

        if aud==1:
            audio_clip=AudioFileClip(path)
            clip_duration = audio_clip.duration
            print(clip_duration)
            clip_run_range=run_range(clip_duration)
            print(clip_run_range)
            for i in range(clip_run_range):
                left=i*time_range
                right=left+time_range
                # print(left,right)
                crop_clip=audio_clip.subclip(left,right)
                try:
                    crop_clip.write_audiofile("vid_to_aud"+str(i)+".mp3")
                except:
                    pass
            
    
    

    # YouTube video URL
    video_url = path
    
    # Create a YouTube object
    yt = YouTube(video_url)
    
    # Get the highest resolution video stream
    stream = yt.streams.get_lowest_resolution()
    
    # Download the video
    stream.download(filename='meeting.mp4')
    
    audio_generator("./meeting.mp4",vid=1)
    
    transcribed_lit=[]

    print("clip run range",clip_run_range)
    for i in tqdm(range(clip_run_range)):
        print("./vid_to_aud"+str(i)+".mp3")
        transcribed=SpeechToTextEng(r"vid_to_aud"+str(i)+".mp3")
        transcribed_lit.append(transcribed)
        os.remove("./vid_to_aud"+str(i)+".mp3")


    data = pd.DataFrame(
        {'transcriptions': transcribed_lit
        })

    # summarizer = pipeline("summarization")
    # joblib.dump(summarizer, 'summarizer.joblib')
    summarizer = joblib.load('summarizer.joblib')
    # sentiment_analyzer = pipeline("sentiment-analysis")
    # joblib.dump(sentiment_analyzer, 'sentiment_analyzer.joblib')
    sentiment_analyzer = joblib.load('sentiment_analyzer.joblib')

    sumarized_lit=[]
    sentiment_lit=[]
    hate_lit=[]
    non_hate_lit=[]

    toxicity_tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
    # toxicity_model = AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
    # joblib.dump(toxicity_model, 'toxicity_model.joblib')
    toxicity_model = joblib.load('toxicity_model.joblib')
    def get_toxicity_score(text):
        inputs = toxicity_tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        outputs = toxicity_model(**inputs)
        predictions = outputs.logits.softmax(dim=1) 
        non_hate,hate=predictions[0].detach().numpy().tolist()   
        return non_hate,hate

    for i in tqdm(range(len(data))):
        summarized=summarizer(data.iloc[i,0],min_length=75, max_length=300)[0]['summary_text']
        sentiment = sentiment_analyzer(data.iloc[i,0])[0]['label']
        non_hate,hate=get_toxicity_score(data.iloc[i,0])
        sumarized_lit.append(summarized)
        sentiment_lit.append(sentiment)
        non_hate_lit.append(non_hate)
        hate_lit.append(hate)

    data['summary']=sumarized_lit
    data['sentiment']=sentiment_lit
    data['non_hate']=non_hate_lit
    data['hate']=hate_lit
    data.to_csv('output.csv', index=False)
    
    tot_text=""
    for i in range(len(data)):
        tot_text=tot_text+data.iloc[i,0]

    # key_model = KeyBERT('distilbert-base-nli-mean-tokens')
    # joblib.dump(key_model, 'key_model.joblib')
    # key_model = joblib.load('key_model.joblib')
    def extract_keywords(text, top_n=20):
        # keywords = key_model.extract_keywords(text, top_n=top_n)
        # return [keyword[0] for keyword in keywords]
        # keywords = WordCloud(width=800, height=400, background_color='white').generate(text)
        # # Get the individual words and their frequencies
        # word_frequencies = keywords.process_text(text)
        # return word_frequencies
        wordcloud = WordCloud(width=800, height=400, background_color='white')
        word_frequencies = wordcloud.process_text(text)

        # Sort the word frequencies and extract the top n words
        sorted_words = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, frequency in sorted_words[:top_n]]

        return top_words
    
    

    


    def summarize_text(text):
        chunk_size = 500  # Number of words per chunk
        total_summary = ""  # Total summary

        words = text.split()  # Split the text into individual words
        num_chunks = len(words) // chunk_size + 1  # Calculate the number of chunks

        for i in tqdm(range(num_chunks)):
            start_index = i * chunk_size
            end_index = start_index + chunk_size
            chunk = " ".join(words[start_index:end_index])

            # Pass the chunk to the summarizer (replace with your summarization code)
            chunk_summary = summarizer(chunk,min_length=75, max_length=200)[0]['summary_text']
            # print(chunk_summary)
            total_summary += chunk_summary

        return total_summary
    tot_keywords=extract_keywords(tot_text)
    tot_summary=summarize_text(tot_text)

    

    non_hate,hate=get_toxicity_score(tot_text)
    os.remove("./meeting.mp4")
    return tot_text,tot_summary,tot_keywords,non_hate,hate




# //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def generate_word_cloud(text):
    # Create a WordCloud object
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    # Display the generated word cloud
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot the word cloud on the axis
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

def toxicity_graph(df):
    # Get the row numbers
    x = df.index
    # Get the values for the 'non_hate' and 'hate' columns
    y1 = df['non_hate']
    y2 = df['hate']
    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot the lines with markers only where there is data
    ax.plot(x, y1, marker='o', linestyle='-', label='non_hate', markevery=[i for i, val in enumerate(y1) if not pd.isnull(val)])
    ax.plot(x, y2, marker='o', linestyle='-', label='hate', markevery=[i for i, val in enumerate(y2) if not pd.isnull(val)])
    # Add labels and title
    ax.set_xlabel('Row Number')
    ax.set_ylabel('Value')
    ax.set_title('Line Graph')
    # Add legend
    ax.legend()
    # Display the plot using Streamlit
    st.pyplot(fig)
def calculate_constructive_criticism(row):
    if row['sentiment'] == 'NEGATIVE' and row['hate'] > row['non_hate']:
        return 0
    elif row['sentiment'] == 'NEGATIVE' and row['non_hate'] > row['hate']:
        return 1
    else:
        return 0
def calculate_criticism(row):
    if row['sentiment'] == 'NEGATIVE' and row['hate'] > row['non_hate']:
        return 1
    elif row['sentiment'] == 'POSITIVE' and row['hate'] > row['non_hate']:
        return 1
    elif row['sentiment'] == 'POSITIVE' and row['non_hate'] > row['hate']:
        return 0
    else:
        return 0
def draw_criticism_graph(df):
    # Get the row numbers
    x = df.index

    # Get the values for the 'constructive_criticism' and 'criticism' columns
    y_constructive = df['constructive_criticism']
    y_criticism = df['criticism']

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the lines
    ax.plot(x, y_constructive, marker='o', label='Constructive Criticism')
    ax.plot(x, y_criticism, marker='o', label='Criticism')

    # Add labels and title
    ax.set_xlabel('Row Number')
    ax.set_ylabel('Value')
    ax.set_title('Constructive Criticism and Criticism Line Plot')

    # Add a legend
    ax.legend()

    # Display the plot using Streamlit
    st.pyplot(fig)

def main():
    st.title("Meeting Summary Web App")

    # YouTube link input
    youtube_url = st.text_input("Enter the YouTube video link")

    if st.button("Process Video"):
        if youtube_url:
            # Process the YouTube video
            tot_text, tot_summary, tot_keywords,non_hate,hate = process_video(youtube_url)

            # Display the output
            if os.path.exists("output2.csv"):
                output_df = pd.read_csv("output.csv")
                st.subheader("Transcriptions:")
                st.write(output_df["transcriptions"])

                # st.subheader("Labels:")
                # st.write(output_df["labels"])

                st.subheader("Word Cloud:")
                generate_word_cloud(tot_text)

                st.subheader("keywords:")
                st.write(tot_keywords)

                st.subheader("tot_text:")
                st.write(tot_text)

                st.subheader("tot_summary:")
                st.write(tot_summary)

                st.subheader("toxity:")
                st.write("non-hate",non_hate)
                st.write("hate",hate)

                new_row = {
                    'transcriptions': "I hate you tihsrah from my gut, you are so useless to the company dont do any work ever.",
                    'summary': "I hate you tihsrah from my gut, you are so useless to the company dont do any work ever.",
                    'sentiment': "NEGATIVE",
                    'non_hate': 0.05,
                    'hate': 0.95,
                    'constructive_criticism': 0
                }

                # Append the new row to the DataFrame
                output_df = output_df.append(new_row, ignore_index=True)
                
                st.subheader("toxity graph:")
                toxicity_graph(output_df)
                output_df['constructive_criticism'] = output_df.apply(calculate_constructive_criticism, axis=1)
                output_df['criticism']=output_df.apply(calculate_criticism, axis=1)

                st.subheader("Critisim Graph:")
                draw_criticism_graph(output_df)


            else:
                st.write("No output file found.")

if __name__ == "__main__":
    main()