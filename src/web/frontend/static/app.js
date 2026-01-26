// Global variable to store detected emotion
let currentEmotion = "Neutral";

window.onload = function() {
  startWebcam();
  // Start analyzing face every 2 seconds
  setInterval(analyzeFace, 2000);
};

async function startWebcam() {
  const video = document.getElementById('webcam');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
  } catch (err) {
    console.error("Error accessing webcam:", err);
    document.getElementById('emotion').textContent = "Cam Error";
  }
}

async function analyzeFace() {
  const video = document.getElementById('webcam');
  if (!video.srcObject) return;

  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  
  const imageData = canvas.toDataURL('image/jpeg');

  try {
    const response = await fetch("/analyze_face", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: imageData })
    });
    
    if (response.ok) {
      const data = await response.json();
      currentEmotion = data.emotion;
      
      const emotionEl = document.getElementById('emotion');
      emotionEl.textContent = currentEmotion;
      
      // Medical-grade emotion color mapping
      const emotionColors = {
        "Angry": "#f87171",    // Muted Red (not pure red)
        "Disgust": "#94a3b8",  // Slate Gray
        "Fear": "#a78bfa",     // Desaturated Purple
        "Happy": "#34d399",    // Soft Green
        "Sad": "#a78bfa",      // Desaturated Purple
        "Surprise": "#fbbf24", // Amber
        "Neutral": "#94a3b8"   // Slate
      };
      
      emotionEl.style.color = emotionColors[currentEmotion] || "#4fd1c5";
    }
  } catch (e) {
    console.error("Face analysis failed", e);
  }
}

// Handle Enter key in input box
function handleKeyPress(event) {
  if (event.key === "Enter") {
    sendMessage();
  }
}

async function sendMessage() {
  const inputEl = document.getElementById("user-input");
  const text = inputEl.value.trim();
  
  if (!text) return;
  
  inputEl.value = ""; // Clear input
  await processUserIntent(text);
}

async function talk() {
  const talkBtn = document.getElementById("talk-btn");

  // Visual Feedback: Listening State
  talkBtn.classList.add("listening");
  talkBtn.disabled = true;

  try {
    // 1. STT: Record and Transcribe
    const sttResponse = await fetch("/stt", { method: "POST" });
    if (!sttResponse.ok) throw new Error("STT failed");
    const sttData = await sttResponse.json();
    const text = sttData.text;

    // Reset Button State
    talkBtn.classList.remove("listening");
    talkBtn.disabled = false;

    if (text) {
      await processUserIntent(text);
    }

  } catch (error) {
    console.error("Error Details:", error);
    
    let errorMessage = "Sorry, I encountered an issue.";
    
    if (error.message.includes("STT failed")) {
      errorMessage = "Microphone/STT Error. Make sure the backend is running.";
    } else if (error.name === 'TypeError' && error.message.includes('fetch')) {
      errorMessage = "Network Error: Could not connect to server.";
    }

    appendMessage("assistant", errorMessage);
    
    // Reset Button State on Error
    talkBtn.classList.remove("listening");
    talkBtn.disabled = false;
  }
}

async function processUserIntent(text) {
  // Display User Message
  appendMessage("user", text);

  try {
    // 2. Consult: Send to LLM
    const response = await fetch("/consult", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        emotion: currentEmotion, 
        text: text
      })
    });

    if (!response.ok) throw new Error("Consultation failed");

    // Prepare Assistant Message Bubble
    const assistantContentNode = appendMessage("assistant", ""); 
    
    // 3. Stream Response
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantText = "";
    let ttsBuffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      assistantText += chunk;
      ttsBuffer += chunk;
      
      assistantContentNode.textContent = assistantText;
      
      // Auto-scroll to bottom of chat
      const chatInterface = document.querySelector(".chat-interface");
      chatInterface.scrollTop = chatInterface.scrollHeight;

      // Check for complete sentences in buffer to stream TTS
      // MOVED TO BACKEND: app.py now handles streaming TTS directly via consult endpoint.
      // We no longer need to send /tts requests from here.
      
      /* 
       * Previously we chunked text here and sent to /tts, but this caused double speaking
       * when combined with server-side streaming TTS.
       */
    } // End of while loop

    
  } catch (error) {
    console.error("Processing Error:", error);
    appendMessage("assistant", "Sorry, I couldn't process your request.");
  }
}

/**
 * Appends a message to the chat container.
 * @param {string} role - 'user' or 'assistant'
 * @param {string} text - Message content
 * @returns {HTMLElement} The text content element (for streaming updates)
 */
function appendMessage(role, text) {
  const chatContainer = document.getElementById("chat");
  
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${role}-message`;
  
  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";
  contentDiv.textContent = text;
  
  messageDiv.appendChild(contentDiv);
  chatContainer.appendChild(messageDiv);
  
  // Scroll to new message
  const chatInterface = document.querySelector(".chat-interface");
  chatInterface.scrollTop = chatInterface.scrollHeight;

  return contentDiv;
}
