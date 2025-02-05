package main

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"time"

	metacall "github.com/metacall/core/source/ports/go_port/source"
)
var (
    batchesStarted   int32 = 0
    batchesCompleted int32 = 0
)
type ProcessMetrics struct {
    Timestamp   time.Time     `json:"timestamp"`
    Operation   string        `json:"operation"`
    Duration    int64         `json:"duration_ms"`
    MemoryUsage uint64        `json:"memory_mb"`
    BatchSize   int          `json:"batch_size,omitempty"`
    WorkerID    int          `json:"worker_id,omitempty"`
}
type BatchMetrics struct {
    BatchNumber    int32         `json:"batch_number"`
    ItemsProcessed int          `json:"items_processed"`
    ProcessingTime int64        `json:"processing_time_ms"`
    MemoryUsage    uint64        `json:"memory_mb"`
    WorkerID       int          `json:"worker_id"`
}
type DetailedResult struct {
    Index int             `json:"index"`
    Text  string         `json:"text"`
    Label string         `json:"label"`
    Score float64        `json:"score"`
    BatchNum int32       `json:"batch_num"`
    WorkerID int         `json:"worker_id"`
    ProcessingTime int64 `json:"processing_time_ms"`
}
type BatchProcessor struct {
    batchSize   int
    numWorkers  int
    inputChan   chan []string
    resultBatchChan chan []DetailedResult
    errorChan   chan error
    metrics     []ProcessMetrics
    batchMetrics []BatchMetrics
    wg          sync.WaitGroup
    totalItems  int
    resultCount int32
    startTime   time.Time
    metricsLock sync.Mutex
}
func getMemoryUsage() uint64 {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    return m.Alloc / 1024 / 1024
}
func NewBatchProcessor(batchSize, numWorkers, totalItems int) *BatchProcessor {
    return &BatchProcessor{
        batchSize:       batchSize,
        numWorkers:      numWorkers,
        inputChan:       make(chan []string, numWorkers*2),
        resultBatchChan: make(chan []DetailedResult, numWorkers*2),
        errorChan:       make(chan error, numWorkers),
        metrics:         make([]ProcessMetrics, 0, totalItems/batchSize+1),
        batchMetrics:    make([]BatchMetrics, 0, totalItems/batchSize+1),
        totalItems:      totalItems,
        resultCount:     0,
        startTime:       time.Now(),
    }
}
func (bp *BatchProcessor) logMetric(operation string, startTime time.Time, extraInfo map[string]interface{}) {
    duration := time.Since(startTime).Milliseconds()
    memory := getMemoryUsage()
    metric := ProcessMetrics{
        Timestamp:   time.Now(),
        Operation:   operation,
        Duration:    duration,
        MemoryUsage: memory,
    }
    if batchSize, ok := extraInfo["batch_size"]; ok {
        metric.BatchSize = batchSize.(int)
    }
    if workerID, ok := extraInfo["worker_id"]; ok {
        metric.WorkerID = workerID.(int)
    }
    bp.metricsLock.Lock()
    bp.metrics = append(bp.metrics, metric)
    bp.metricsLock.Unlock()
    fmt.Printf("%s [Worker %d] completed in %dms (Memory: %dMB)\n", 
        operation, metric.WorkerID, duration, memory)
}
func (bp *BatchProcessor) worker(id int) {
    defer bp.wg.Done()
    workerStart := time.Now()
    for batch := range bp.inputChan {
        if err := bp.processBatch(batch, id); err != nil {
            bp.errorChan <- fmt.Errorf("worker %d: %v", id, err)
        }
    }
    bp.logMetric("Worker_Complete", workerStart, map[string]interface{}{
        "worker_id": id,
    })
}
func (bp *BatchProcessor) processBatch(batch []string, workerID int) error {
    batchStart := time.Now()
    batchNum := atomic.AddInt32(&batchesStarted, 1)
    fmt.Printf("Worker %d processing batch %d (size: %d)\n", workerID, batchNum, len(batch))
    result, err := metacall.Call("process_batch", batch)
    if err != nil {
        return fmt.Errorf("metacall error: %v", err)
    }
    results, ok := result.([]interface{})
    if !ok {
        return fmt.Errorf("unexpected result type: %T", result)
    }
    batchResults := make([]DetailedResult, 0, len(results))
    processTime := time.Since(batchStart).Milliseconds()
    for i, r := range results {
        if mapResult, ok := r.(map[string]interface{}); ok {
            detailedResult := DetailedResult{
                Index:         (int(batchNum)-1)*bp.batchSize + i,
                Text:         batch[i],
                Label:        mapResult["label"].(string),
                Score:        mapResult["score"].(float64),
                BatchNum:     batchNum,
                WorkerID:     workerID,
                ProcessingTime: processTime,
            }
            batchResults = append(batchResults, detailedResult)
        }
    }
    bp.resultBatchChan <- batchResults
    atomic.AddInt32(&bp.resultCount, int32(len(batchResults)))
    memory := getMemoryUsage()
    bp.metricsLock.Lock()
    bp.batchMetrics = append(bp.batchMetrics, BatchMetrics{
        BatchNumber:    batchNum,
        ItemsProcessed: len(batch),
        ProcessingTime: processTime,
        MemoryUsage:    memory,
        WorkerID:       workerID,
    })
    bp.metricsLock.Unlock()
    fmt.Printf("Batch %d completed by worker %d in %dms (Memory: %dMB)\n", 
        batchNum, workerID, processTime, memory)
    atomic.AddInt32(&batchesCompleted, 1)
    return nil
}
func writeDetailedResults(results []DetailedResult, filename string) error {
    output := struct {
        Metadata struct {
            TotalRecords    int       `json:"total_records"`
            NegativeCount   int       `json:"negative_count"`
            PositiveCount   int       `json:"positive_count"`
            ProcessingTime  int64     `json:"processing_time_ms"`
            AverageBatchTime int64    `json:"average_batch_time_ms"`
            TotalBatches    int32     `json:"total_batches"`
        } `json:"metadata"`
        Results []DetailedResult `json:"results"`
    }{
        Results: results,
    }
    for _, r := range results {
        if r.Label == "NEGATIVE" {
            output.Metadata.NegativeCount++
        } else {
            output.Metadata.PositiveCount++
        }
    }
    output.Metadata.TotalRecords = len(results)
    output.Metadata.TotalBatches = atomic.LoadInt32(&batchesCompleted)
    jsonData, err := json.MarshalIndent(output, "", "  ")
    if err != nil {
        return err
    }
    return os.WriteFile(filename, jsonData, 0644)
}
func loadCSV(filename string) ([]string, error) {
    file, err := os.Open(filename)
    if err != nil {
        return nil, fmt.Errorf("error opening file: %v", err)
    }
    defer file.Close()
    reader := csv.NewReader(file)
    headers, err := reader.Read()
    if err != nil {
        return nil, fmt.Errorf("error reading header: %v", err)
    }
    descriptionIndex := -1
    for i, header := range headers {
        if header == "description" {
            descriptionIndex = i
            break
        }
    }
    if descriptionIndex == -1 {
        return nil, fmt.Errorf("description column not found in CSV")
    }
    descriptions := make([]string, 0, 1000) 
    records, err := reader.ReadAll()
    if err != nil {
        return nil, fmt.Errorf("error reading records: %v", err)
    }
    for i, record := range records {
        if i >= 1000 {
            break
        }
        if descriptionIndex < len(record) {
            descriptions = append(descriptions, record[descriptionIndex])
        } else {
            return nil, fmt.Errorf("record %d doesn't have enough columns", i)
        }
    }
    return descriptions, nil
}
func main() {
    runtime.GOMAXPROCS(runtime.NumCPU())
    totalStart := time.Now()
    initStart := time.Now()
    if err := metacall.Initialize(); err != nil {
        log.Fatalf("Failed to initialize MetaCall: %v", err)
    }
    fmt.Printf("MetaCall initialization took %dms\n", time.Since(initStart).Milliseconds())
    scriptStart := time.Now()
    if err := metacall.LoadFromFile("py", []string{"ml_processor.py"}); err != nil {
        log.Fatalf("Failed to load Python script: %v", err)
    }
    fmt.Printf("Script loading took %dms\n", time.Since(scriptStart).Milliseconds())
    csvStart := time.Now()
    descriptions, err := loadCSV("wine-reviews.csv")
    if err != nil {
        log.Fatalf("Failed to load CSV: %v", err)
    }
    fmt.Printf("CSV loading took %dms (loaded %d descriptions)\n", 
        time.Since(csvStart).Milliseconds(), len(descriptions))
    optimalBatchSize := 32 
    numWorkers := runtime.NumCPU() 
    bp := NewBatchProcessor(optimalBatchSize, numWorkers, len(descriptions))
    processingStart := time.Now()
    batches := make([][]string, 0, (len(descriptions)+optimalBatchSize-1)/optimalBatchSize)
    currentBatch := make([]string, 0, optimalBatchSize)
    for _, desc := range descriptions {
        currentBatch = append(currentBatch, desc)
        if len(currentBatch) == optimalBatchSize {
            batches = append(batches, currentBatch)
            currentBatch = make([]string, 0, optimalBatchSize)
        }
    }
    if len(currentBatch) > 0 {
        batches = append(batches, currentBatch)
    }
    var results []DetailedResult
    done := make(chan bool)
    go func() {
        for batchResult := range bp.resultBatchChan {
            results = append(results, batchResult...)
            fmt.Printf("Progress: %d/%d items processed\n", len(results), len(descriptions))
        }
        done <- true
    }()
    bp.wg.Add(numWorkers)
    for i := 0; i < numWorkers; i++ {
        go bp.worker(i)
    }
    for _, batch := range batches {
        bp.inputChan <- batch
    }
    close(bp.inputChan)
    bp.wg.Wait()
    close(bp.resultBatchChan)
    <-done
    totalTime := time.Since(totalStart).Milliseconds()
    processingTime := time.Since(processingStart).Milliseconds()
    negative_count := 0
    positive_count := 0
    for _, result := range results {
        if result.Label == "NEGATIVE" {
            negative_count++
        } else {
            positive_count++
        }
    }
    if err := os.MkdirAll("results", 0755); err != nil {
        log.Printf("Failed to create results directory: %v", err)
    }
    timestamp := time.Now().Format("20060102_150405")
    resultFile := fmt.Sprintf("results/detailed_results_%s.json", timestamp)
    if err := writeDetailedResults(results, resultFile); err != nil {
        log.Printf("Failed to write results: %v", err)
    }
    metricsFile := fmt.Sprintf("results/process_metrics_%s.json", timestamp)
    metricsJSON, _ := json.MarshalIndent(bp.metrics, "", "  ")
    os.WriteFile(metricsFile, metricsJSON, 0644)
    batchMetricsFile := fmt.Sprintf("results/batch_metrics_%s.json", timestamp)
    batchMetricsJSON, _ := json.MarshalIndent(bp.batchMetrics, "", "  ")
    os.WriteFile(batchMetricsFile, batchMetricsJSON, 0644)
    fmt.Printf("\nExecution Summary:\n")
    fmt.Printf("================\n")
    fmt.Printf("Total Records: %d\n", len(results))
    fmt.Printf("Negative Results: %d\n", negative_count)
    fmt.Printf("Positive Results: %d\n", positive_count)
    fmt.Printf("Total Time: %dms\n", totalTime)
    fmt.Printf("Processing Time: %dms\n", processingTime)
    fmt.Printf("Average Time per Item: %.2fms\n", float64(processingTime)/float64(len(results)))
    fmt.Printf("Batch Size: %d\n", optimalBatchSize)
    fmt.Printf("Worker Count: %d\n", numWorkers)
    fmt.Printf("Total Batches: %d\n", len(batches))
    fmt.Printf("Final Memory Usage: %dMB\n", getMemoryUsage())
    fmt.Printf("\nResults and metrics saved in results/ directory\n")
}