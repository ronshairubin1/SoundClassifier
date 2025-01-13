<!DOCTYPE html>
<html>
<head>
    <title>Inference Statistics</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        table { border-collapse: collapse; width: 50%; }
        td, th { border: 1px solid #ddd; padding: 8px; }
        th { background-color: #f4f4f4; }
    </style>
</head>
<body>
    <h1>Inference Statistics</h1>
    <p><strong>Total Predictions Made:</strong> {{ inference_stats['total_predictions'] }}</p>
    <p><strong>Average Confidence Level:</strong> {{ avg_confidence | round(4) }}</p>
    
    <h2>Class Counts</h2>
    <table>
        <tr><th>Class Name</th><th>Count</th></tr>
        {% for class_name, count in inference_stats['class_counts'].items() %}
        <tr>
            <td>{{ class_name }}</td>
            <td>{{ count }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
