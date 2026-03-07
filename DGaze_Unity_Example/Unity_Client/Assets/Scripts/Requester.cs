using AsyncIO;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;
using System.Globalization;
using System.Text.RegularExpressions;


public class Requester : RunAbleThread
{
    public string recordingsString;
    static readonly Regex FloatRegex = new Regex(@"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan", RegexOptions.IgnoreCase | RegexOptions.Compiled);

    protected override void Run()
    {
        ForceDotNet.Force(); // this line is needed to prevent unity freeze after one use, not sure why yet
        using (RequestSocket client = new RequestSocket())
        {
            client.Connect("tcp://localhost:5555");

            while (Running)
            {
                if (recordingsString != null)
                {
                    client.SendFrame(recordingsString);
                    // ReceiveFrameString() blocks the thread until you receive the string, but TryReceiveFrameString()
                    // do not block the thread, you can try commenting one and see what the other does, try to reason why
                    // unity freezes when you use ReceiveFrameString() and play and stop the scene without running the server
                    //                string message = client.ReceiveFrameString();
                    //                Debug.Log("Received: " + message);
                    string message = null;
                    bool gotMessage = false;
                    while (Running)
                    {
                        gotMessage = client.TryReceiveFrameString(out message); // this returns true if it's successful
                        if (gotMessage) break;
                    }

                    if (gotMessage)
                    {
                        UpdatePredictedGaze(message);
                        Debug.Log("On-Screen Gaze Position: " + message);
                    }
                }
            }
        }

        NetMQConfig.Cleanup(); // this line is needed to prevent unity freeze after one use, not sure why yet
    }

    void UpdatePredictedGaze(string message)
    {
        MatchCollection matches = FloatRegex.Matches(message ?? "");
        if (matches.Count < 2)
        {
            GazeDebugState.ClearPredictedGaze();
            return;
        }

        bool okX = TryParseFiniteFloat(matches[0].Value, out float gazeX);
        bool okY = TryParseFiniteFloat(matches[1].Value, out float gazeY);
        if (!okX || !okY)
        {
            GazeDebugState.ClearPredictedGaze();
            return;
        }

        GazeDebugState.SetPredictedGaze(new Vector2(gazeX, gazeY));
    }

    bool TryParseFiniteFloat(string text, out float value)
    {
        if (!float.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out value))
            return false;
        if (float.IsNaN(value) || float.IsInfinity(value))
            return false;
        return true;
    }
}
