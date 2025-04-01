import React, { useEffect, useRef, useState } from 'react';
import Webcam from 'react-webcam';

function App() {
  const webcamRef = useRef(null);
  const [recognition, setRecognition] = useState({ user: '', emotion: '' });
  const [isBackendConnected, setIsBackendConnected] = useState(true);

  const captureAndSendFrame = async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        try {
          // Fetch User Identity
          const userResponse = await fetch("http://localhost:5000/identify", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: imageSrc }), // Key is 'image'
          });

          // Fetch Emotion
          const emotionResponse = await fetch("http://localhost:5000/emotion", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: imageSrc }), // Key is 'image'
          });

          if (!userResponse.ok || !emotionResponse.ok) {
            throw new Error(
              `HTTP error! Status codes: Identify(${userResponse.status}), Emotion(${emotionResponse.status})`
            );
          }

          const userData = await userResponse.json();
          const emotionData = await emotionResponse.json();

          setRecognition({
            user: userData.User,
            emotion: emotionData.Mood,
          });
          setIsBackendConnected(true);
        } catch (error) {
          console.error("Error sending frame:", error);
          setIsBackendConnected(false);
        }
      }
    }
  };

  useEffect(() => {
    // Send frames to the backend at regular intervals (100ms for real-time experience)
    const interval = setInterval(captureAndSendFrame, 500); // Increased to 500ms to reduce load
    return () => clearInterval(interval);
  }, []);

  if (!isBackendConnected) {
    return (
      <div>
        <h1>Backend Connection Error</h1>
        <p>Unable to connect to the facial recognition service.</p>
        <p>Please ensure the backend server is running at http://localhost:5000</p>
      </div>
    );
  }

  return (
    <div>
      <Webcam ref={webcamRef} screenshotFormat="image/jpeg" />
      <div>
        <h2>Current Mood</h2>
        <p>{recognition.emotion || 'Analyzing...'}</p>
        <h2>Identified User</h2>
        <p>{recognition.user || 'Scanning...'}</p>
      </div>
    </div>
  );
}

export default App;
