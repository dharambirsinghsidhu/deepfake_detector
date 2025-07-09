# A CNN and Entropy-Based Hybrid Model for Robust Deepfake Detection

<br>

## 📄 Overview

This project presents a novel hybrid deepfake detection model that integrates **Convolutional Neural Networks (CNNs)** with **entropy-based randomness analysis**. Designed to overcome the limitations of existing CNN-only models, especially against high-quality, fine-grained deepfakes, our approach significantly enhances classification reliability and reduces false positives.

Deepfake technology, while having legitimate uses in fields like education and entertainment, has been increasingly exploited for malicious activities such as **misinformation, identity fraud, blackmail, and online scams**. Traditional CNN-based detectors often struggle with advanced deepfakes due to their reliance solely on visual cues and their susceptibility to adversarial attacks. 

Our hybrid model addresses this by combining the robust spatial feature extraction capabilities of DenseNet-121 with the statistical regularities identified through Shannon entropy, capturing inconsistencies inherent in GAN-generated images.

---

<br>

## 🚀 Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes.

### Prerequisites

Requirements for the software and other tools to build, test and push 
- Python 3.8+ (preferably 3.11)

### Installation

1.  **Clone the Repository:**
   
    ```bash
    git clone https://github.com/dharambirsinghsidhu/EntropyVision_Deepfake_Detector.git
    cd EntropyVision_Deepfake_Detector
    ```

3.  **Create and Activate a Virtual Environment:**

    It's highly recommended to use a virtual environment to manage project dependencies.

    ```bash
    python -m venv venv
    ```

    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Python Dependencies:**

    First, ensure your `pip` installer is up to date, then install the required Python libraries.

    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    pip install streamlit
    ```

End with an example of getting some data out of the system or using it
for a little demo

---

<br>

## ▶️ Running the Application

Once everything is installed, starting the app is simple:

1.  Go to your project's main folder in your terminal.
2.  Type this command and press Enter:
    ```bash
    streamlit run app2.py --server.enableCORS false --server.enableXsrfProtection false
    ```
3.  A new tab will automatically open in your web browser showing the application, usually at `http://localhost:8501`.

---

<br>

## 💡 Using the Deepfake Detector (EntropyVision)

Our tool is designed for easy deepfake detection:

1.  Start the Streamlit application in your browser.
2.  **Upload any image** you want to verify.
3.  The system will process it and show you whether it's **real or a deepfake**, complete with a **confidence level**.


### Style test

Checks if the best practices and the right coding style has been used.

    Give an example

## Deployment

Add additional notes to deploy this on a live system

## Built With

  - [Contributor Covenant](https://www.contributor-covenant.org/) - Used
    for the Code of Conduct
  - [Creative Commons](https://creativecommons.org/) - Used to choose
    the license

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions
available, see the [tags on this
repository](https://github.com/PurpleBooth/a-good-readme-template/tags).

## Authors

  - **Billie Thompson** - *Provided README Template* -
    [PurpleBooth](https://github.com/PurpleBooth)

See also the list of
[contributors](https://github.com/PurpleBooth/a-good-readme-template/contributors)
who participated in this project.

## License

This project is licensed under the [CC0 1.0 Universal](LICENSE.md)
Creative Commons License - see the [LICENSE.md](LICENSE.md) file for
details

## Acknowledgments

  - Hat tip to anyone whose code is used
  - Inspiration
  - etc
