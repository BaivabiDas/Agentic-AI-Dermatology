
$results = @()

$folders = Get-ChildItem ".\test" -Directory

foreach ($folder in $folders) {

    $image = Get-ChildItem $folder.FullName -File |
        Where-Object { $_.Extension -match '\.(jpg|jpeg|png|bmp|webp)$' } |
        Select-Object -First 1

    if ($null -eq $image) {
        Write-Host "No image found in $($folder.Name)"
        continue
    }

    Write-Host "Testing: $($folder.Name)"

    $response = curl.exe -s -X POST "http://localhost:5001/predict" `
        -F "image=@$($image.FullName)" | ConvertFrom-Json

    $results += [PSCustomObject]@{
        ActualDisease    = $folder.Name
        PredictedDisease = $response.disease
        Confidence       = $response.confidence_percent
        PredictionTime   = $response.prediction_time_seconds
        Status           = if ($response.disease -eq $folder.Name) {"Correct"} else {"Incorrect"}
    }
}

$results | Export-Csv ".\class_test_results.csv" -NoTypeInformation

$results | Format-Table -AutoSize

$correct = @($results | Where-Object {$_.Status -eq "Correct"}).Count
$total = $results.Count

Write-Host "`nCorrect predictions: $correct / $total"
Write-Host "Accuracy: $([math]::Round(($correct / $total) * 100, 2))%"
Write-Host "Report saved: class_test_results.csv"
