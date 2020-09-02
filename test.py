<script>
var ctx = document.getElementById('chart1').getContext('2d');

const verticalLinePlugin = {
        getLinePosition: function (chart, pointIndex) {
            const meta = chart.getDatasetMeta(0); // first dataset is used to discover X coordinate of a point
            const data = meta.data;
            return data[pointIndex]._model.x;
        },
        renderVerticalLine: function (chartInstance, pointIndex) {
            const lineLeftOffset = this.getLinePosition(chartInstance, pointIndex);
            const scale = chartInstance.scales['y-axis-0'];
            const context = chartInstance.chart.ctx;

            // render vertical line
            context.beginPath();
            context.strokeStyle = '#ff0000';
            context.moveTo(lineLeftOffset, scale.top);
            context.lineTo(lineLeftOffset, scale.bottom);
            context.stroke();

            // write label
            context.fillStyle = "#ff0000";
            context.textAlign = 'center';
            context.font = '15px Helvetica';
            context.fillText('FIRE DETECTED', lineLeftOffset - scale.right, (scale.bottom - scale.top) / 2 + .25*scale.top);    
        },

        afterDatasetsDraw: function (chart, easing) {
            if (chart.config.lineAtIndex) {
                chart.config.lineAtIndex.forEach(pointIndex => this.renderVerticalLine(chart, pointIndex));
            }
        }
    };

Chart.plugins.register(verticalLinePlugin);
Chart.defaults.global.defaultFontColor='rgba(255, 255, 255, 1)'
var config = {
    type: 'line',
    data: {
        labels: [{% for item in time_pts %}'{{item}}',{% endfor %}],
        datasets: [
        {
            label: 'Alarm Threshold',
            data: [
                {% for pt in cloud_bound %}
                    {
                    t: new Date("{{pt.0}}"),
                    y: {{ pt.1 }}
                    },
                {% endfor %}
            ],
            yAxisID: 'B',
            backgroundColor: [
                'rgba(0, 0, 0, 0)',
            ],
            borderColor: [
                'rgba(220, 227, 223, 1)',
            ],
            borderWidth: 1,
            borderDash: [10,5],
            steppedLine: true,
        },
        {
            label: 'Alarm Level',
            data: [
                {% for pt in diff_pts %}
                    {
                    t: new Date("{{pt.0}}"),
                    y: {{ pt.1 }}
                    },
                {% endfor %}
            ],
            yAxisID: 'B',
            backgroundColor: [
                'rgba(0, 0, 0, 0)',
            ],
            borderColor: [
                'rgba(253, 155, 160, 1)',
            ],
            pointBorderColor: [{% for item in cloud_colors %} '{{item}}',{% endfor %}],
            borderWidth: 4,
        },
        {
            label: 'Predicted Near Infared',
            data: [
                {% for pt in pred_pts %}
                    {
                    t: new Date("{{pt.0}}"),
                    y: {{ pt.1 }}
                    },
                {% endfor %}
            ],
            backgroundColor: [
                'rgba(0, 253, 220, 0.3)',
            ],
            borderColor: [
                'rgba(0, 253, 220, 1)',
            ],
            borderWidth: 3
        }, 
        {
            label: 'Actual Near Infared',
            data: [
                {% for pt in actual_7_pts %}
                    {
                    t: new Date("{{pt.0}}"),
                    y: {{ pt.1 }}
                    },
                {% endfor %}
            ],
            backgroundColor: [
                'rgba(220, 0, 115, 0.3)',
            ],
            borderColor: [
                'rgba(255, 0, 170, 1)',
            ],
            borderWidth: 3
        },]
    },
    options: {
        responsive: true,
        title: {
            display: true,
            fontSize: 30,
            text: 'What the Artificial Intelligence Sees'
        },
        tooltips: {
            mode: 'index',
        },
        hover: {
            mode: 'index'
        },
        scales: {
            xAxes: [{
                scaleLabel: {
                    display: true,
                }
            }],
            yAxes: [{
                stacked: false,
                type: 'linear',
                position: 'left',
                scaleLabel: {
                    display: true,
                    labelString: 'Near Infared Radiance',
                    fontColor: 'rgba(71, 103, 120, 1)',
                    fontSize: 20,
                },
            },
            {
                stacked: false,
                type: 'linear',
                position: 'right',
                id: 'B',
                scaleLabel: {
                    display: true,
                    labelString: 'Alarm Level',
                    fontColor: 'rgba(253, 155, 160, 1)',
                    fontSize: 20, 
                },
            },]
        }
    },
    lineAtIndex: [{{ fire_start_idx }}],
}
var chart1 = new Chart(ctx, config);
</script>