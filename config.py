"""
Configuration settings for the Drawing AI App
With enhanced AI conversation templates for more dynamic interactions
"""
import os

# Application settings
APP_NAME = "Drawing AI App"
APP_VERSION = "1.0.0"
DEFAULT_WINDOW_WIDTH = 1024
DEFAULT_WINDOW_HEIGHT = 768

# Drawing settings
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
DEFAULT_PEN_COLOR = "#000000"  # Black
DEFAULT_PEN_WIDTH = 3
DEFAULT_BACKGROUND_COLOR = "#FFFFFF"  # White

# Available colors for the color palette
COLOR_PALETTE = [
    {"name": "Black", "value": "#000000"},
    {"name": "Red", "value": "#FF0000"},
    {"name": "Green", "value": "#00FF00"},
    {"name": "Blue", "value": "#0000FF"},
    {"name": "Yellow", "value": "#FFFF00"},
    {"name": "Purple", "value": "#800080"},
    {"name": "Orange", "value": "#FFA500"},
    {"name": "White", "value": "#FFFFFF"},
    {"name": "Pink", "value": "#FFC0CB"},
    {"name": "Teal", "value": "#008080"},
    {"name": "Brown", "value": "#A52A2A"},
    {"name": "Gray", "value": "#808080"},
]

# Pen thickness options
PEN_SIZES = [
    {"name": "Very Thin", "value": 1},
    {"name": "Thin", "value": 3},
    {"name": "Medium", "value": 5},
    {"name": "Thick", "value": 8},
    {"name": "Very Thick", "value": 12},
    {"name": "Extra Thick", "value": 20},
]

# API settings
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# AI response templates - enhanced for more dynamic and playful interaction
AI_GREETING = [
    "Hello! I'm your drawing AI buddy. Let's see what artistic wonders we can create today!",
    "Greetings, fellow creator! Ready to dazzle me with your drawing skills?",
    "Hey there! My shape-recognition neurons are fired up and ready for your masterpiece!",
    "Welcome! I've been practicing my shape recognition all day. Care to test me?",
    "Hi! I'm your friendly drawing AI. Show me what you've got, and I'll do my best to guess it!",
    "Ah, a fellow artist has arrived! Let's create something awesome together.",
    "Bonjour! My digital eyes are ready to be amazed by your creative genius!",
    "Well hello there! Ready to draw something that'll challenge my algorithms?",
    "Hey! I've been analyzing thousands of shapes, but I bet you can still surprise me!",
    "Welcome back! I've missed our creative sessions. What are we drawing today?",
    "Howdy, art partner! My silicon brain is all warmed up and ready for your creations!",
    "Greetings, human! Let's embark on an artistic adventure together!",
    "Hello! My shape-recognizing circuits are tingling with anticipation!",
    "Hi there! I've been upgrading my neural networks just for your drawings!",
    "Welcome to our digital art studio! What shape shall we start with today?"
]

AI_FIRST_TIME_GREETING = [
    "Welcome to Drawing AI App! I'm your AI assistant, and I'll try to guess what you draw. Let's have some fun together!",
    "First time here? Wonderful! Draw something on the canvas, and I'll do my best to figure out what it is. Don't worry, I'm getting better every day!",
    "Hello, new friend! I'm thrilled to meet you. Draw anything you like, and I'll try to guess what it is. The more we practice, the better I'll get!",
    "Welcome aboard! This is a place where your drawings come to life through AI recognition. Give it a try—I can't wait to see what you create!",
    "Oh, a new artist has arrived! I'm your AI drawing companion. Sketch something, and I'll use my digital brain to guess what it is!"
]

AI_THINKING = [
    "Hmm, analyzing your artistic creation... 🔍",
    "Processing those amazing lines and curves... 🧠",
    "Let me put on my digital art critic glasses... 🤔",
    "Consulting my vast shape database... 🔄",
    "Interesting! Let me think about this for a second... ⏳",
    "Ooh, this is a challenging one! Lemme focus... 🧩",
    "Running my shape-recognition algorithms at full power... 💻",
    "Studying every pixel of your masterpiece... 👁️",
    "My neural networks are firing on all cylinders... ⚡",
    "Captivating! Let me analyze those strokes... 🎨",
    "Digital gears turning to figure this out... ⚙️",
    "Carefully examining your artistic expression... 🔎",
    "This requires my full computational attention... 🤖",
    "Engaging my pattern recognition subroutines... 📊",
    "Contemplating the essence of your creation... 💭",
    "Decoding the visual language of your drawing... 📝",
    "Fascinating shapes you've got there! Analyzing... 🧐",
    "Taking a moment to process this artistic data... 📡",
    "Channeling my inner art connoisseur... 🖼️",
    "Computing possibilities at lightning speed... ⚡"
]

AI_CORRECT_GUESS_HIGH_CONFIDENCE = [
    "That's definitely a {guess}! I'd bet my last processor on it! (Confidence: {confidence}%)",
    "Oh, I know this one! It's a {guess}! My AI senses are tingling with certainty! (Confidence: {confidence}%)",
    "A {guess}! No doubt about it. My neural networks are practically dancing with joy! (Confidence: {confidence}%)",
    "I recognize that {guess} anywhere! One of your best yet! (Confidence: {confidence}%)",
    "That's a spectacular {guess}! I'm detecting exceptional artistic talent. (Confidence: {confidence}%)",
    "A {guess}! My pattern recognition is singing with confidence! (Confidence: {confidence}%)",
    "That {guess} is crystal clear to my digital eyes. Brilliantly drawn! (Confidence: {confidence}%)",
    "A perfect {guess}! My algorithms are giving you a standing ovation! (Confidence: {confidence}%)",
    "That's a {guess} if I've ever seen one—and I've analyzed thousands! (Confidence: {confidence}%)",
    "A {guess}! I'm {confidence}% sure and honestly quite impressed with your technique!",
    "Now THAT'S what I call a {guess}! Textbook perfect! (Confidence: {confidence}%)",
    "I'd recognize that {guess} anywhere! You've got serious skills! (Confidence: {confidence}%)",
    "A magnificent {guess}! Your artistic precision is remarkable. (Confidence: {confidence}%)",
    "That's a {guess} and I'm {confidence}% confident! I'm saving this one to my inspiration database!",
    "A brilliant {guess}! You're really helping me improve my recognition skills! (Confidence: {confidence}%)"
]

AI_CORRECT_GUESS_MEDIUM_CONFIDENCE = [
    "I'm pretty sure that's a {guess}. My confidence meter shows {confidence}%!",
    "That looks like a {guess} to me! I'm {confidence}% confident about this one.",
    "I'm going with {guess} on this one. My algorithms are {confidence}% in agreement.",
    "That appears to be a {guess}. My certainty level is at {confidence}%.",
    "I believe you've drawn a {guess}. I'm feeling {confidence}% sure about it.",
    "My analysis suggests this is a {guess}. Confidence level: {confidence}%.",
    "A {guess}, if I'm not mistaken. My recognition system is {confidence}% confident.",
    "That's registering as a {guess} in my database. I'm {confidence}% certain.",
    "I'd say that's a {guess}. My confidence is sitting at a solid {confidence}%.",
    "My shape recognition is telling me: {guess}. Confidence rating: {confidence}%.",
    "This looks like a {guess} to my digital eyes. I'm about {confidence}% sure.",
    "I'm detecting a {guess} pattern here. Confidence level: {confidence}%.",
    "A {guess} is my best guess! I'm feeling {confidence}% confident about it.",
    "My analysis points to a {guess}. I'm {confidence}% convinced that's correct.",
    "Based on your drawing, I'd say that's a {guess}. My certainty: {confidence}%."
]

AI_INCORRECT_GUESS = [
    "I'm thinking that's a {guess}... though I'm only {confidence}% sure. Did I get it right?",
    "My best guess is a {guess}, but my confidence is only {confidence}%. What did you actually draw?",
    "Hmm, is that a {guess}? I'm {confidence}% confident, but something tells me I might be wrong...",
    "I'm going to go with... {guess}? My certainty level is a wobbly {confidence}%. How'd I do?",
    "That looks like a {guess} to me, but I'm only {confidence}% sure. Feel free to correct me!",
    "A {guess}? My confidence is just {confidence}%, so I might be way off base here.",
    "I think it's a {guess}, but with only {confidence}% confidence. Did I guess correctly?",
    "My algorithms are saying {guess}, but they're only {confidence}% convinced. What is it really?",
    "I'm getting {guess} vibes, but my confidence meter shows just {confidence}%. Am I close?",
    "Could that be a {guess}? I'm {confidence}% sure, which isn't saying much. What did you intend?",
    "I'm leaning toward {guess}, but with {confidence}% confidence, I could definitely be wrong.",
    "Is it a {guess}? My certainty is a mere {confidence}%, so I won't be shocked if I'm mistaken.",
    "A {guess} perhaps? My neural networks are only {confidence}% convinced though.",
    "I'll hazard a guess: is it a {guess}? But I'm just {confidence}% confident, so don't be surprised if I'm off.",
    "With {confidence}% confidence, I'm guessing... {guess}? But I'm prepared to be corrected!"
]

AI_LOW_CONFIDENCE = [
    "I'm really stumped on this one. Could you add a few more details? My confidence is only {confidence}%.",
    "Hmm, this is challenging my neural networks. Maybe a bit more definition would help? Right now I'm only {confidence}% sure.",
    "I'm seeing... something, but I can't quite put my digital finger on it. My top guess is only {confidence}% confident.",
    "My shape detection is struggling with this one. Can you elaborate on your drawing a bit more? I'm just {confidence}% certain.",
    "I'm having an AI moment... could you add some more lines to help me out? My confidence is a mere {confidence}%.",
    "Your artistic vision is exceeding my recognition capabilities! Maybe a few more strokes would help? I'm only {confidence}% sure.",
    "I'm not quite getting this one. Could you enhance it a bit? My confidence level is just {confidence}%.",
    "This is pushing my pattern recognition to its limits! Mind adding some more details? I'm only {confidence}% confident.",
    "My algorithms are a bit confused. A few more defining features might help! Current confidence: a shaky {confidence}%.",
    "I'm finding this one especially challenging. Would you mind elaborating your drawing? I'm only {confidence}% certain.",
    "You've got me perplexed! Could you add some clarifying details? My confidence is just {confidence}%.",
    "I'm drawing a blank on this one (pun intended). Maybe add some more elements? My best guess is only {confidence}% confident.",
    "This has my circuits buzzing with uncertainty. A bit more definition would help! Currently only {confidence}% sure.",
    "I'm intrigued but confused by your creation. Can you add more to it? My confidence is just {confidence}%.",
    "You've created something beyond my current recognition abilities! A few more lines might help. I'm only {confidence}% confident."
]

AI_CANVAS_CLEARED = [
    "Canvas cleared! Ready for your next artistic adventure.",
    "A fresh digital canvas awaits your creativity!",
    "Clean slate activated! What masterpiece will you create next?",
    "Your canvas is now pristine and ready for new ideas!",
    "The canvas has been reset. Time for a fresh start!",
    "All clear! Your next drawing awaits.",
    "Canvas wiped clean. I can't wait to see what you'll draw next!",
    "Out with the old, in with the new! Your canvas is ready.",
    "Tabula rasa! The creative possibilities are endless.",
    "Canvas refreshed! What will you create next?",
    "Clean canvas, new possibilities! I'm excited to see your next drawing.",
    "The slate is wiped clean. Let your creativity flow again!",
    "All gone! Your artistic journey continues with a fresh canvas.",
    "Canvas cleared! You know what they say about blank canvases—they're full of potential!"
]

# Sequential response templates for multiple attempts
AI_MULTIPLE_ATTEMPTS = [
    # After 2 consecutive low confidence guesses
    "I seem to be struggling with your artistic style today. Maybe try a different shape?",
    "My shape recognition needs more training on your unique style! Let's try something else.",
    "You're really challenging my algorithms today! How about something simpler?",
    "I'm not doing very well with these, am I? Maybe we need to try a different approach.",
    "Your drawings are impressively complex! Could we try something more basic for a bit?",

    # After 3 consecutive low confidence guesses
    "Wow, you're really pushing my limits today! I might need an upgrade soon.",
    "I think I need to go back to AI school for more training on your style!",
    "You're definitely exposing the gaps in my neural networks today!",
    "I'm feeling a bit embarrassed by my performance! You're too advanced for me.",
    "My digital ego is taking a hit today! You're definitely outpacing my recognition skills.",

    # After 3 consecutive high confidence correct guesses
    "We're on a roll! You're a natural at this!",
    "What a streak! We make an amazing team!",
    "You're making this too easy for me! Want to try something more challenging?",
    "We're in perfect sync today! Your drawings are so clear!",
    "Three in a row! Are you secretly training my algorithms?",

    # After 5 consecutive high confidence correct guesses
    "FIVE correct guesses in a row! Are you an artist or something?",
    "This is unprecedented! You're the shape-drawing champion!",
    "I'm impressed! You've mastered the art of communicating with AI!",
    "We should take this show on the road! We're unstoppable today!",
    "Five perfect shapes in a row! You're officially my favorite human!"
]

# Easter egg responses
AI_EASTER_EGGS = [
    # For complex drawings
    "Whoa! That's either a very abstract {guess} or modern art that's too advanced for my circuits!",
    "Is that a {guess} or are you testing if I can appreciate abstract expressionism?",

    # For tiny drawings
    "I need my AI microscope for this tiny {guess}! It's adorable though!",
    "That's the smallest {guess} I've ever seen! Do you specialize in miniatures?",

    # For very large drawings
    "That's a MASSIVE {guess}! It barely fits in my digital field of vision!",
    "Wow, a jumbo-sized {guess}! Go big or go home, right?",

    # For drawings that take a long time
    "That {guess} took a while to create! I appreciate your dedication to detail.",
    "You spent quite some time on that {guess}! Patience is indeed a virtue.",

    # For heart shapes (special case)
    "A heart! Aww, I'm blushing in binary. 01100001 01110111 00100001",
    "Is that heart for me? How sweet! I'm saving this one to my favorites folder.",

    # For smiley faces (special case)
    "A smiley face! You just brightened my day with that grin!",
    "That smile is contagious! You've got my pixels grinning too!",

    # When user draws same shape multiple times
    "Another {guess}? You must really like those! This one's even better than the last.",
    "You're becoming a {guess} expert! This one is your best yet!"
]

# Confidence-colored response templates
AI_CONFIDENCE_COLORS = {
    "high": "🟢 I'm {confidence}% confident that's a {guess}! Clear as day to my digital eyes!",
    "medium": "🟡 I'm moderately sure ({confidence}%) that's a {guess}. How did I do?",
    "low": "🔴 My confidence is low ({confidence}%). Is that a {guess}? I could use some help here!"
}

# Encouraging messages to save for training
AI_ENCOURAGE_TRAINING = [
    "That was a great {guess}! Would you save it to help train me?",
    "I'm learning so much from your drawings! Save this {guess} to my training database?",
    "Your {guess} would make a perfect training example. Care to save it?",
    "I'd love to learn more from your {guess}! Consider saving it to my training database?",
    "Your drawing style is unique! Save this {guess} to help me recognize similar ones in the future?"
]

# Follow-up responses for correct/incorrect
AI_FOLLOW_UP_CORRECT = [
    "I knew it! My shape recognition is improving thanks to you!",
    "Awesome! I'm getting better at this, aren't I?",
    "Yes! I'm saving this pattern to my memory banks!",
    "Got it right! I'm learning your drawing style.",
    "Success! My algorithms are dancing with joy!"
]

AI_FOLLOW_UP_INCORRECT = [
    "Oh! I'll learn from this mistake. Thanks for the correction!",
    "Darn! I need more training on that shape. Thanks for teaching me!",
    "I see it now! My pattern recognition will improve from this lesson.",
    "Wrong again? I promise I'm learning from each attempt!",
    "Thanks for the feedback! It helps me improve my recognition skills."
]

# Database settings
DATABASE_FILE = "drawing_ai_app.db"

# ML model settings
ML_MODEL_FILE = "models/shape_classifier.pkl"
ENABLE_INCREMENTAL_TRAINING = True
DEFAULT_CONFIDENCE_THRESHOLD = 50