from transformers import pipeline

# Load the text generation model
generator = pipeline('text-generation', model='gpt2')

# Function to split the paragraph into smaller chunks
def chunk_text(paragraph, chunk_size=1024):
    return [paragraph[i:i + chunk_size] for i in range(0, len(paragraph), chunk_size)]

# Function to explain each chunk of the paragraph
def explain_paragraph(paragraph):
    chunks = chunk_text(paragraph)  # Split paragraph into chunks of max 1024 tokens
    explanations = []  # To store explanations for each chunk
    
    for chunk in chunks:
        # More specific prompt to guide the model to generate an explanation
        prompt = f"Provide a detailed explanation or summary of the following text: {chunk}"
        
        # Generate the explanation
        explanation = generator(
            prompt, 
            max_new_tokens=150,  # Generate up to 150 new tokens for each chunk
            num_return_sequences=1,  # Return a single sequence
            do_sample=True,  # Enable sampling for more creative responses
            temperature=0.7,  # Adjusts randomness (lower values are more deterministic)
            top_p=0.9, # Nucleus sampling to generate diverse results
            clean_up_tokenization_spaces=True 
        )
        
        # Append the generated explanation (not the input text) to the list
        explanations.append(explanation[0]['generated_text'].replace(prompt, '').strip())
    
    return " ".join(explanations)  # Combine explanations from all chunks

# Example paragraph (replace this with any large text)
paragraph = """2 Nature of electromagnetic waves
It can be shown from Maxwell’s equations that electric
and magnetic fields in an electromagnetic wave are
perpendicular to each other, and to the direction of
propagation. It appears reasonable, say from our
discussion of the displacement current. Consider
Fig. 8.2. The electric field inside the plates of the capacitor
is directed perpendicular to the plates. The magnetic
field this gives rise to via the displacement current is
along the perimeter of a circle parallel to the capacitor
plates. So B and E are perpendicular in this case. This
is a general feature.
In Fig. 8.3, we show a typical example of a plane
electromagnetic wave propagating along the z direction
(the fields are shown as a function of the z coordinate, at
a given time t). The electric field Ex
 is along the x-axis,
and varies sinusoidally with z, at a given time. The
magnetic field B
y
 is along the y-axis, and again varies
sinusoidally with z. The electric and magnetic fields Ex
and B
y
 are perpendicular to each
other, and to the direction z of
propagation. We can write Ex
 and
B
y
 as follows:
Ex
= E0
 sin (kz–wt) [8.7(a)]
B
y
= B0
 sin (kz–wt)
"""

# Call the explain_paragraph function and print the explanation
explanation = explain_paragraph(paragraph)
print(explanation)
