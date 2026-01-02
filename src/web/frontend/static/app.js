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
      
      // Update color based on emotion
      const emotionColors = {
        "Angry": "#ef4444",   // Red
        "Disgust": "#84cc16", // Lime Green
        "Fear": "#a855f7",    // Purple
        "Happy": "#22c55e",   // Green
        "Sad": "#3b82f6",     // Blue
        "Surprise": "#eab308",// Yellow
        "Neutral": "#94a3b8"  // Gray
      };
      
      emotionEl.style.color = emotionColors[currentEmotion] || "var(--accent-color)";
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
  const talkIcon = talkBtn.querySelector(".icon");

  // Visual Feedback: Listening State
  talkBtn.classList.add("listening");
  talkIcon.textContent = "👂";
  talkBtn.disabled = true;

  try {
    // 1. STT: Record and Transcribe
    const sttResponse = await fetch("/stt", { method: "POST" });
    if (!sttResponse.ok) throw new Error("STT failed");
    const sttData = await sttResponse.json();
    const text = sttData.text;

    // Reset Button State
    talkBtn.classList.remove("listening");
    talkIcon.textContent = "🎙️";
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
    talkIcon.textContent = "🎙️";
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
      // Match sentences ending with . ! ? or newline
      let match = ttsBuffer.match(/.+?([.!?\n]|$)\s*/); 
      // Actually, we want to be careful not to match partials if checks "|$" at end of string which is always true.
      // Better regex: match any sequence ending in punctuation.
      // We look for content followed by punctuation, non-greedy.
      // But we must assume if we don't find punctuation, we wait for next chunk. 
      // So we only strip if we find punctuation.
      
      let sentenceMatch = ttsBuffer.match(/(.*?[.!?\n])\s+/);
      // The \s+ ensures we have a break after the sentence (or end of chunk space), 
      // helping avoid splitting "Mr. Smith" if "Mr." was matched? No, simplistic is robust enough for now.
      // Let's stick to simple: split by [.!?\n]
      
      // Let's use a simpler loop with index checking
      let puncIndex = -1;
      const puncs = ['.', '!', '?', '\n'];
      
      // Find first punctuation
      for(let p of puncs) {
          let idx = ttsBuffer.indexOf(p);
          if (idx !== -1 && (puncIndex === -1 || idx < puncIndex)) {
             puncIndex = idx;
          }
      }

      if (puncIndex !== -1) {
          // We found a sentence end. 
          // Extract it including the punctuation.
          // Wait, if "Mr." case? 
          // For a simple assistant, "stop at dot" is 90% okay.
          
          // Actually, let's use a regex loop to find all minimal sentences
          while (true) {
              const result = ttsBuffer.match(/(.*?[.!?\n])/);
              if (!result) break;
              
              const sentence = result[0];
              // Send to TTS (Fire & Forget)
              if (sentence.trim().length > 0) {
                 fetch("/tts", {
                   method: "POST",
                   headers: { "Content-Type": "application/json" },
                   body: JSON.stringify({ text: sentence })
                 }).catch(e => console.error("TTS Error", e));
              }

              // Remove from buffer
              ttsBuffer = ttsBuffer.substring(result.index + sentence.length);
          }
      }
    }
    
    // Flush remaining buffer
    if (ttsBuffer.trim().length > 0) {
        fetch("/tts", {
           method: "POST",
           headers: { "Content-Type": "application/json" },
           body: JSON.stringify({ text: ttsBuffer })
        }).catch(e => console.error("TTS Error", e));
    }
    
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
