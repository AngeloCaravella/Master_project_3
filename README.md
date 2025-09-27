# EV Charging Experiment Interface

This is a web-based user interface for simulating EV charging station experiments.

## How to Run This Application

**IMPORTANT:** You cannot run this application by simply opening the `index.html` file in your browser (`file://...`). Modern web applications require a web server to function correctly due to browser security policies (CORS).

You must serve the files from a local web server. Here are two simple ways to do this:

### Option 1: Using Python (Recommended if you have Python installed)

1.  Open your terminal or command prompt.
2.  Navigate to the directory containing the `index.html` file.
3.  Run the following command:

    ```bash
    # If you have Python 3
    python3 -m http.server

    # If you have Python 2
    python -m SimpleHTTPServer
    ```
4.  The server will start. Open your web browser and go to the address it shows, usually `http://localhost:8000`.

### Option 2: Using Node.js / npx

1.  Make sure you have Node.js installed.
2.  Open your terminal or command prompt.
3.  Navigate to the directory containing the `index.html` file.
4.  Run the following command:

    ```bash
    npx serve
    ```
5.  The server will start. Open your web browser and go to the address it shows, usually `http://localhost:3000`.

Now you should see the application interface and be able to use it to load your custom scenario files.
