# Sample Video Resources

This folder contains resources for testing and evaluating the AI Smart Surveillance System.

## Quick Generation / Download

To generate a synthetic surveillance scenario video (offline, zero-network required):
```bash
python samples/download_or_generate_sample.py
```
This generates `samples/sample_surveillance.mp4` featuring:
- Floor grid and marked restricted perimeter zone
- Walking pedestrian entering and triggering Zone Intrusion
- Loitering stationary for >10 seconds
- Abandoning a stationary backpack/luggage
- Additional pedestrians testing crowding logic

To download a sample video from the public Intel IoT sample repository:
```bash
python samples/download_or_generate_sample.py --download
```

## Running the Pipeline on the Sample Video

Using CLI:
```bash
python main.py --source samples/sample_surveillance.mp4
```

Using Dashboard:
```bash
streamlit run app.py
```
Select "Sample Video / File" in the sidebar.
