const original_trans = document.getElementById('original_transcript');
        const detected_trans = document.getElementById('detection');
        const corrected_trans = document.getElementById('correction');
        const recording_prompt = document.getElementById('recording_prompt');
        const messageContainer = document.getElementById('upload_message');
        const fileInput = document.getElementById('file')
        const startBtn = document.getElementById('startButton');
        const stopBtn = document.getElementById('stopButton');
        const eventSource = new EventSource('/stream');

        function startRecording(){
            recording_prompt.innerHTML = "Recording started."
            fetch('/start_recording')
                .then(response => response.json())
                .then(data => {
                    // recording_prompt.innerHTML = data.message
                })
                .catch(error => {
                    recording_prompt.innerHTML = error
                });
            startBtn.disabled = true;
            stopBtn.disabled = false;
        }

        function stopRecording(){
            recording_prompt.innerHTML = "Recording Stopped."
            fetch('/stop_recording')
                .then(response => response.json())
                .then(data => {
                    recording_prompt.innerHTML = `${data.message}. Duration: ${data.duration}seconds`
                    original_trans.innerHTML += `${data.transcription} `
                })
                .catch(error => {
                    recording_prompt.innerHTML = error
                });
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }

        async function uploadAudio(event){
            event.preventDefault();  // prevent the default form submission
            
             // Get the form and its data
             const form = event.target;
            const formData = new FormData(form);
            
            messageContainer.textContent = 'Uploading File and reading the transcripts...'

            try {
                // Make a POST request to upload the audio
                const response = await fetch(form.action, {
                    method: 'POST',
                    body: formData
                });

                // Read the response text
                const transcripts = await response.text();

                let message = 'File Uploaded Succesfully!';

                // Change the color based on response
                if (response.ok) {
                    messageContainer.textContent = message;
                    messageContainer.style.color = 'green'; // Success message
                    original_trans.innerHTML = transcripts
                } else {
                    messageContainer.textContent = `Error code ${response.status}`
                    messageContainer.style.color = 'red'; // Error message
                }
            } catch (error) {
                // Handle errors (e.g., network issues)
                console.error('Error uploading audio:', error);
                messageContainer.textContent = 'An error occurred while uploading. Please try again.';
                messageContainer.style.color = 'red';
            }
        }


        // eventSource.onmessage = function(event) {
        //     const data = event.data;
        //     detected_trans.innerHTML += data
        // };

        eventSource.onerror = function(error) {
            console.error("Error in SSE connection: ", error);
        };

        function reset(){
            recording_prompt.innerHTML = ""
            original_trans.innerHTML = ""
            detected_trans.innerHTML = ""
            corrected_trans.innerHTML = ""
            fileInput.value = ""
            messageContainer.textContent = ""
        }