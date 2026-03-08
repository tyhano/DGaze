using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

// Collect DGaze_ET training features to CSV.
// One row contains: timestamp, screen-gaze, angular-gaze, head velocity, and tracked object features.
public class DGazeETDataCollector : MonoBehaviour
{
    public GameObject DynamicObjects;
    public Camera headCamera;

    // Sampling frequency (Hz)
    public float sampleRate = 100f;
    // Optional offset to calibrate actual sample interval (ms)
    public float timeOffsetMs = 0f;

    // Demo gaze source: mouse position normalized to [0,1]
    public bool useMouseAsGaze = true;

    // Output location: <Application.persistentDataPath>/DGazeRecords/<fileName>
    public string fileName = "dgaze_et_frames.csv";

    bool running;
    StreamWriter writer;
    TrackObjects trackObjects;
    CalculateHeadVelocity headVelocity;

    readonly CultureInfo invariantCulture = CultureInfo.InvariantCulture;

    void Start()
    {
        if (sampleRate <= 0)
            sampleRate = 100f;

        running = true;

        if (headCamera != null)
            headVelocity = headCamera.GetComponent<CalculateHeadVelocity>();

        if (DynamicObjects != null)
            trackObjects = DynamicObjects.GetComponent<TrackObjects>();

        string recordDir = Path.Combine(Application.persistentDataPath, "DGazeRecords");
        Directory.CreateDirectory(recordDir);
        string outputPath = Path.Combine(recordDir, fileName);

        writer = new StreamWriter(outputPath, false);
        writer.AutoFlush = true;
        writer.WriteLine(BuildHeader());

        Debug.Log("DGazeETDataCollector writing to: " + outputPath);
        StartCoroutine(RecordData());
    }

    IEnumerator RecordData()
    {
        float waitSeconds = 1f / sampleRate - timeOffsetMs / 1000f;
        if (waitSeconds <= 0f)
            waitSeconds = 1f / sampleRate;

        WaitForSecondsRealtime waitTime = new WaitForSecondsRealtime(waitSeconds);
        while (running)
        {
            WriteOneFrame();
            yield return waitTime;
        }
    }

    void WriteOneFrame()
    {
        long timestampMs = GetUnixTimeMs();

        Vector2 screenGaze = GetScreenGaze();
        Coordinate angularGaze = ScreenCoord2AngularCoord(new Coordinate { posX = screenGaze.x, posY = screenGaze.y });

        float headVelX = 0f;
        float headVelY = 0f;
        if (headVelocity != null)
        {
            headVelX = SanitizeFloat(headVelocity.headVelX);
            headVelY = SanitizeFloat(headVelocity.headVelY);
        }

        float[] objectFeatures = GetObjectFeatures(); // 9 values for 3 objects

        string row = string.Join(",", new string[]
        {
            timestampMs.ToString(),
            F(screenGaze.x),
            F(screenGaze.y),
            F(angularGaze.posX),
            F(angularGaze.posY),
            F(headVelX),
            F(headVelY),
            F(objectFeatures[0]), F(objectFeatures[1]), F(objectFeatures[2]),
            F(objectFeatures[3]), F(objectFeatures[4]), F(objectFeatures[5]),
            F(objectFeatures[6]), F(objectFeatures[7]), F(objectFeatures[8])
        });

        writer.WriteLine(row);
    }

    Vector2 GetScreenGaze()
    {
        if (useMouseAsGaze)
        {
            Vector3 mousePos = Input.mousePosition;
            float gazeX = Mathf.Clamp01(SanitizeFloat(mousePos.x / Screen.width));
            float gazeY = Mathf.Clamp01(SanitizeFloat(mousePos.y / Screen.height));
            return new Vector2(gazeX, gazeY);
        }

        // Placeholder for eye-tracker integration.
        return new Vector2(0.5f, 0.5f);
    }

    float[] GetObjectFeatures()
    {
        float[] features = new float[9];
        if (trackObjects == null || string.IsNullOrEmpty(trackObjects.trackedObjectsString))
            return features;

        string[] tokens = trackObjects.trackedObjectsString.Split(',');
        int limit = Mathf.Min(tokens.Length, features.Length);
        for (int i = 0; i < limit; ++i)
        {
            if (float.TryParse(tokens[i], NumberStyles.Float, invariantCulture, out float value))
                features[i] = SanitizeFloat(value);
        }

        return features;
    }

    string BuildHeader()
    {
        return "timestamp_ms,gaze_screen_x,gaze_screen_y,gaze_angle_x,gaze_angle_y,head_vel_x,head_vel_y,"
             + "obj1_angle_x,obj1_angle_y,obj1_dist,obj2_angle_x,obj2_angle_y,obj2_dist,obj3_angle_x,obj3_angle_y,obj3_dist";
    }

    string F(float value)
    {
        return SanitizeFloat(value).ToString("f6", invariantCulture);
    }

    long GetUnixTimeMs()
    {
        TimeSpan timeSpan = DateTime.Now - new DateTime(1970, 1, 1, 0, 0, 0);
        return (long)timeSpan.TotalMilliseconds - 8 * 60 * 60 * 1000;
    }

    float SanitizeFloat(float value)
    {
        if (float.IsNaN(value) || float.IsInfinity(value))
            return 0f;
        return value;
    }

    struct Coordinate
    {
        public float posX;
        public float posY;
    }

    // Transform normalized screen coordinates to angular coordinates.
    Coordinate ScreenCoord2AngularCoord(Coordinate screenCoord)
    {
        // HTC Vive parameters used by the original project
        float verticalFov = Mathf.PI * 110f / 180f;
        float screenWidth = 1080f;
        float screenHeight = 1200f;
        float screenCenterX = 0.5f * screenWidth;
        float screenCenterY = 0.5f * screenHeight;
        float screenDist = 0.5f * screenHeight / Mathf.Tan(verticalFov / 2f);

        screenCoord.posX *= screenWidth;
        screenCoord.posY *= screenHeight;

        Coordinate angularCoord;
        angularCoord.posX = Mathf.Atan((screenCoord.posX - screenCenterX) / screenDist) / Mathf.PI * 180f;
        angularCoord.posY = Mathf.Atan((screenCoord.posY - screenCenterY) / screenDist) / Mathf.PI * 180f;
        return angularCoord;
    }

    void OnDestroy()
    {
        running = false;
        if (writer != null)
        {
            writer.Flush();
            writer.Close();
            writer = null;
        }
    }
}
