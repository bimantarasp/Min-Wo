// Load the TensorFlow.js model
const MODEL_URL = 'tfjs_model/model.json';
const modelPromise = tf.loadLayersModel(MODEL_URL);

// Helper function to preprocess user input
function preprocessInput(inputText) {
  const sanitizedInput = inputText
    .toLowerCase()
    .replace(/[^\w\s]|_/g, '')
    .replace(/\s+/g, ' ');
  return [sanitizedInput];
}

// Helper function to display chat log
function displayChatLog(message) {
  const chatLog = document.getElementById('chat-log');
  const messageElement = document.createElement('div');
  messageElement.textContent = message;
  chatLog.appendChild(messageElement);
}

// Handle user input
async function handleUserInput() {
  const userInput = document.getElementById('user-input').value;
  const preprocessedInput = preprocessInput(userInput);

  // Load the model
  const model = await modelPromise;

  // Preprocess the input
  const input = tokenizer.textsToSequences(preprocessedInput);
  const paddedInput = padSequences(input, { maxlen: input_shape });

  // Make predictions
  const output = model.predict(paddedInput);
  const predictedClass = tf.argMax(output, axis=1).dataSync()[0];
  const responseTag = label.classes_[predictedClass];
  const response = random.choice(response[responseTag]);

  // Display the response
  displayChatLog('Min-Wo: ' + response);

  // Clear the input field
  document.getElementById('user-input').value = '';

  // Check if the conversation should end
  if (responseTag === 'goodbye') {
    return;
  }
}

// Attach event listener to send button
document.getElementById('send-button').addEventListener('click', handleUserInput);
