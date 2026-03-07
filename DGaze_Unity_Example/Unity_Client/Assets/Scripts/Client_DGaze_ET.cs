using UnityEngine;
using System;
using System.Text;
using UnityEngine.UI;


public class Client_DGaze_ET : MonoBehaviour
{
    string recordingsString;
    Requester requester;
    public GameObject dataRecorder;

    public bool showCurrentGazeCross = true;
    public bool showPredictedGazeCross = true;
    public float crossSizePx = 14f;
    public float lineThicknessPx = 2f;

    Texture2D lineTexture;

    private void Start()
    {
        requester = new Requester();
        requester.recordingsString = null;
        requester.Start();
        recordingsString = null;

        lineTexture = new Texture2D(1, 1, TextureFormat.RGBA32, false);
        lineTexture.SetPixel(0, 0, Color.white);
        lineTexture.Apply();

    }


    // Update is called once per frame
    void Update()
    {
        recordingsString = dataRecorder.GetComponent<DataRecorder_DGaze_ET>().recordingsString;
        if (recordingsString != null)
            requester.recordingsString = recordingsString;
    }

    void OnGUI()
    {
        GazeDebugState.GetSnapshot(out Vector2 current, out Vector2 predicted, out bool hasPredicted);

        if (showCurrentGazeCross)
            DrawCross(current, Color.red);

        if (showPredictedGazeCross && hasPredicted)
            DrawCross(predicted, Color.blue);
    }

    void DrawCross(Vector2 normalizedPos, Color color)
    {
        if (lineTexture == null)
            return;

        float x = Mathf.Clamp01(normalizedPos.x) * Screen.width;
        float y = (1f - Mathf.Clamp01(normalizedPos.y)) * Screen.height;

        Rect horizontal = new Rect(x - crossSizePx, y - lineThicknessPx * 0.5f, crossSizePx * 2f, lineThicknessPx);
        Rect vertical = new Rect(x - lineThicknessPx * 0.5f, y - crossSizePx, lineThicknessPx, crossSizePx * 2f);

        Color lastColor = GUI.color;
        GUI.color = color;
        GUI.DrawTexture(horizontal, lineTexture);
        GUI.DrawTexture(vertical, lineTexture);
        GUI.color = lastColor;
    }


    private void OnDestroy()
    {
        requester.Stop();
        if (lineTexture != null)
            Destroy(lineTexture);
    }
}
