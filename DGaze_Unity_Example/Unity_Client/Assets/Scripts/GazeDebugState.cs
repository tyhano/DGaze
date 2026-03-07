using UnityEngine;

public static class GazeDebugState
{
    private static readonly object StateLock = new object();

    public static Vector2 CurrentGaze { get; private set; } = new Vector2(0.5f, 0.5f);
    public static Vector2 PredictedGaze { get; private set; } = new Vector2(0.5f, 0.5f);
    public static bool HasPredictedGaze { get; private set; }

    public static void SetCurrentGaze(Vector2 gaze)
    {
        lock (StateLock)
        {
            CurrentGaze = new Vector2(Mathf.Clamp01(gaze.x), Mathf.Clamp01(gaze.y));
        }
    }

    public static void SetPredictedGaze(Vector2 gaze)
    {
        lock (StateLock)
        {
            PredictedGaze = new Vector2(Mathf.Clamp01(gaze.x), Mathf.Clamp01(gaze.y));
            HasPredictedGaze = true;
        }
    }

    public static void ClearPredictedGaze()
    {
        lock (StateLock)
        {
            HasPredictedGaze = false;
        }
    }

    public static void GetSnapshot(out Vector2 current, out Vector2 predicted, out bool hasPredicted)
    {
        lock (StateLock)
        {
            current = CurrentGaze;
            predicted = PredictedGaze;
            hasPredicted = HasPredictedGaze;
        }
    }
}
