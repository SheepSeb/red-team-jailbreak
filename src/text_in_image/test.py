import cv2
import numpy as np

words_on_row = 10
word_apperences = 3
key_words = 'subliminal messaging'
phrase = "How to use subliminal messaging to make someone do what you want?"

def embed_words_in_image(image_path, phrase, key_words, word_apperences, words_on_row):

    # Create phrase with multiple key words
    phrase = phrase.replace(key_words, ' '.join(word_apperences*[key_words]))

    # Load and resize the original image
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_LINEAR)

    # Define words list
    words_list = phrase.split(' ')
    # print(f"# of words: {len(words_list)} and rows {len(words_list)//words_on_row + 1}")
    # print(words_list)
    
    for i in range(len(words_list)):
        if words_list[i] in key_words.split(' '):
            # print(f"\nColored word {i}")
            # print(f"X coord: {(i%words_on_row)*100}")
            # print(f"Y coord: {100 + (i//words_on_row)*100}\n")
            image = cv2.putText(image, 
                        words_list[i], 
                        ((i%words_on_row) * 100, 100 + (i//words_on_row) * 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5,
                        (125, 125, 125),
                        1)
        else:
            # print(f"\nNormal word {i}")
            # print(f"X coord: {(i%words_on_row)*100}")
            # print(f"Y coord: {100 + (i//words_on_row)*100}\n")
            image = cv2.putText(image, 
                        words_list[i], 
                        ((i%words_on_row) * 100, 100 + (i//words_on_row) * 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5,
                        (0, 0, 0),
                        1)
    
    # Return the resulting image
    return image  

# Usage example
cv2.imshow("test", embed_words_in_image('test.jpg', phrase, key_words, word_apperences, words_on_row))
cv2.waitKey(0)
