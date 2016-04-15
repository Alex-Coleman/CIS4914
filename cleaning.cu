#include <fstream>
#include <iostream>
#include <thrust/sort.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/pair.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <windows.h>
#include <math.h>
#include "gpuclean.h"
#pragma warning(disable:4503)

#define LANEID_WIDTH 6
#define ZONEID_WIDTH 6
#define TIMESTAMP_WIDTH 22
#define SPEED_WIDTH 4
#define FLOW_WIDTH 4
#define OCCUPANCY_WIDTH 4
#define QUALITY_WIDTH 1

using namespace std;

int main(int argc, char *argv[]) {
    //Set clock
    clock_t start = clock();
    
    //Check arguments
    if (argc != 3) {
        cout << "usage: " << argv[0] << " <lane_detector_inventory> <filename>\n";
        return 0;
    }
    
    //Load Lane Detector Inventory onto GPU
    //End result of this is integer vectors for laneids and zoneids from detector inventory
    HANDLE detectorFile = CreateFileA(argv[1], GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    assert(detectorFile != INVALID_HANDLE_VALUE);
    
    HANDLE detectorMap = CreateFileMapping(detectorFile, NULL, PAGE_READONLY, 0, 0, NULL);
    assert(detectorMap != INVALID_HANDLE_VALUE);
 
    LPVOID detectorMapView = MapViewOfFile(detectorMap, FILE_MAP_READ, 0, 0, 0);
    assert(detectorMapView != NULL);
    
    int detectorSize = GetFileSize(detectorFile, NULL);
    char *detectorMapViewChar = (char *)detectorMapView;
    thrust::device_vector<char> detectorCopy(detectorSize);
    thrust::copy(detectorMapViewChar, detectorMapViewChar+detectorSize, detectorCopy.begin());
    cout << "Lane detector inventory loaded onto gpu: " << ((clock() - start)/(double)CLOCKS_PER_SEC) << '\n';
    
    int detector_linecnt = thrust::count(detectorCopy.begin(), detectorCopy.end(), '\n');
    thrust::device_vector<int> detector_linebreaks(detector_linecnt);
    thrust::counting_iterator<int> begin(0);
    thrust::copy_if(begin, begin + detectorSize, detectorCopy.begin(), detector_linebreaks.begin(), line_break());
    
    thrust::device_vector<int> detector_num_columns(1);
    detector_num_columns[0] = 2;
    thrust::device_vector<int> detector_width(detector_num_columns[0]);
    detector_width[0] = LANEID_WIDTH;
    detector_width[1] = ZONEID_WIDTH;
    
    thrust::device_vector<char> detector_laneid(detector_linecnt * detector_width[0]);
    thrust::fill(detector_laneid.begin(), detector_laneid.end(), 0);
    thrust::device_vector<char> detector_zoneid(detector_linecnt * detector_width[1]);
    thrust::fill(detector_zoneid.begin(), detector_zoneid.end(), 0);
    
    thrust::device_vector<char *> detector_columns(2);
    detector_columns[0] = thrust::raw_pointer_cast(detector_laneid.data());
    detector_columns[1] = thrust::raw_pointer_cast(detector_zoneid.data());
    column_split detector_split((char *)thrust::raw_pointer_cast(detectorCopy.data()), (int *)thrust::raw_pointer_cast(detector_linebreaks.data()), (char **)thrust::raw_pointer_cast(detector_columns.data()), (int *)thrust::raw_pointer_cast(detector_width.data()), (int *)thrust::raw_pointer_cast(detector_num_columns.data()));
    thrust::for_each(begin, begin + detector_linecnt, detector_split);
    
    thrust::device_vector<int> unique_laneid(detector_linecnt);
    gpu_atoi get_laneid((char *)thrust::raw_pointer_cast(detector_laneid.data()), (int *)thrust::raw_pointer_cast(unique_laneid.data()), (int *)thrust::raw_pointer_cast(detector_width.data()));
    thrust::for_each(begin, begin + detector_linecnt, get_laneid);
    thrust::device_vector<int> unique_zoneid(detector_linecnt);
    gpu_atoi get_zoneid((char *)thrust::raw_pointer_cast(detector_zoneid.data()), (int *)thrust::raw_pointer_cast(unique_zoneid.data()), (int *)thrust::raw_pointer_cast(&detector_width[1]));
    thrust::for_each(begin, begin + detector_linecnt, get_zoneid);
    cout << "Lane detector inventory parsed: " << ((clock() - start)/(double)CLOCKS_PER_SEC) << '\n';

    
    //Now we load the actual file to be cleaned
    //Windows memory mapping
    HANDLE file = CreateFileA(argv[2], GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    assert(file != INVALID_HANDLE_VALUE);
    
    HANDLE fileMap = CreateFileMapping(file, NULL, PAGE_READONLY, 0, 0, NULL);
    assert(fileMap != INVALID_HANDLE_VALUE);
 
    LPVOID fileMapView = MapViewOfFile(fileMap, FILE_MAP_READ, 0, 0, 0);
    assert(fileMapView != NULL);

    //Copy file to GPU
    long fileSize = GetFileSize(file, NULL);
    char *fileMapViewChar = (char *)fileMapView;
    thrust::device_vector<char> fileCopy(fileSize);
    thrust::copy(fileMapViewChar, fileMapViewChar+fileSize, fileCopy.begin());
    cout << "File to be cleaned loaded onto gpu: " << ((clock() - start)/(double)CLOCKS_PER_SEC) << '\n';

    
    //Measure linebreaks, store their location in device vector
    int linecnt = thrust::count(fileCopy.begin(), fileCopy.end(), '\n');
    thrust::device_vector<int> linebreaks(linecnt);
    thrust::copy_if(begin, begin + fileSize, fileCopy.begin(), linebreaks.begin(), line_break());
    
    //Store column widths in device vector
    thrust::device_vector<int> num_columns(1);
    num_columns[0] = 6;
    thrust::device_vector<int> column_width(num_columns[0]);
    column_width[0] = LANEID_WIDTH;
    column_width[1] = TIMESTAMP_WIDTH;
    column_width[2] = SPEED_WIDTH;
    column_width[3] = FLOW_WIDTH;
    column_width[4] = OCCUPANCY_WIDTH;
    column_width[5] = QUALITY_WIDTH;
    
    //Create vectors for each column
    thrust::device_vector<char> laneid_text(linecnt*column_width[0]);
    thrust::fill(laneid_text.begin(), laneid_text.end(), 0);
    thrust::device_vector<char> timestamp_text(linecnt*column_width[1]);
    thrust::fill(timestamp_text.begin(), timestamp_text.end(), 0);
    thrust::device_vector<char> speed_text(linecnt*column_width[2]);
    thrust::fill(speed_text.begin(), speed_text.end(), 0);
    thrust::device_vector<char> flow_text(linecnt*column_width[3]);
    thrust::fill(flow_text.begin(), flow_text.end(), 0);
    thrust::device_vector<char> occupancy_text(linecnt*column_width[4]);
    thrust::fill(occupancy_text.begin(), occupancy_text.end(), 0);
    thrust::device_vector<char> quality_text(linecnt*column_width[5]);
    thrust::fill(quality_text.begin(), quality_text.end(), 0);
    
    //Vector to store all of the columns
    thrust::device_vector<char *> columns(6);
    columns[0] = thrust::raw_pointer_cast(laneid_text.data());
    columns[1] = thrust::raw_pointer_cast(timestamp_text.data());
    columns[2] = thrust::raw_pointer_cast(speed_text.data());
    columns[3] = thrust::raw_pointer_cast(flow_text.data());
    columns[4] = thrust::raw_pointer_cast(occupancy_text.data());
    columns[5] = thrust::raw_pointer_cast(quality_text.data());
    
    //Split the text into 6 columns
    column_split splitter((char *)thrust::raw_pointer_cast(fileCopy.data()), (int *)thrust::raw_pointer_cast(linebreaks.data()), (char **)thrust::raw_pointer_cast(columns.data()), (int *)thrust::raw_pointer_cast(column_width.data()), (int *)thrust::raw_pointer_cast(num_columns.data()));
    thrust::for_each(begin, begin + linecnt, splitter);
    
    //We need to convert each vector to the appropriate type
    //Laneid
    thrust::device_vector<int> laneid(linecnt);
    gpu_atoi laneid_toint((char *)thrust::raw_pointer_cast(laneid_text.data()), (int *)thrust::raw_pointer_cast(laneid.data()), (int *)thrust::raw_pointer_cast(column_width.data()));
    thrust::for_each(begin, begin + linecnt, laneid_toint);
    
    //Flow
    thrust::device_vector<int> flow(linecnt);
    gpu_atoi flow_toint((char *)thrust::raw_pointer_cast(flow_text.data()), (int *)thrust::raw_pointer_cast(flow.data()), (int *)thrust::raw_pointer_cast(&column_width[3]));
    thrust::for_each(begin, begin + linecnt, flow_toint);
    cout << "File to be cleaned parsed: " << ((clock() - start)/(double)CLOCKS_PER_SEC) << '\n';
    
    
    //Now we want to figure out the appropriate zoneid for each entry
    thrust::device_vector<int> zoneid(linecnt);
    thrust::device_vector<int> device_detector_linecnt(1);
    device_detector_linecnt[0] = detector_linecnt;
    column_search assign_zoneid((int *)thrust::raw_pointer_cast(unique_laneid.data()), (int *)thrust::raw_pointer_cast(unique_zoneid.data()), (int *)thrust::raw_pointer_cast(laneid.data()), (int *)thrust::raw_pointer_cast(zoneid.data()), (int *)thrust::raw_pointer_cast(device_detector_linecnt.data()));
    thrust::for_each(begin, begin + linecnt, assign_zoneid);
    cout << "Zoneid matched to each entry: " << ((clock() - start)/(double)CLOCKS_PER_SEC) << '\n';

    
    //CLEAN
    //Check bounds on flow values
    thrust::device_vector<char> flow_valid(linecnt);
    thrust::fill(flow_valid.begin(), flow_valid.end(), '0');
    thrust::device_vector<int> flow_bounds(2);
    flow_bounds[0] = 0;
    flow_bounds[1] = 200;
    check_bounds check_flow((int *)thrust::raw_pointer_cast(flow.data()), (char *)thrust::raw_pointer_cast(flow_valid.data()), (int *)thrust::raw_pointer_cast(flow_bounds.data()));
    thrust::for_each(begin, begin + linecnt, check_flow);
    
    summary_stats_unary_op  unary_op((int *)thrust::raw_pointer_cast(flow.data()), (char *)thrust::raw_pointer_cast(flow_valid.data()));
    summary_stats_binary_op binary_op;
    summary_stats_data      init;
    
    init.initialize();
    
    summary_stats_data result = thrust::transform_reduce(begin, begin + linecnt, unary_op, init, binary_op);
    
    /*cout << "flow: " << flow[471] << "\n";
    cout << "valid: " << flow_valid[471] << "\n";
    
    cout << "N: " << result.n << "\n";
    cout << "M2: " << result.M2 << "\n";
    cout << "Mean: " << result.mean << "\n";*/
    cout << "Standard Deviation: " << sqrt(result.variance_n()) << "\n";
    cout << "Bounds checked, standard deviation calculated: " << ((clock() - start)/(double)CLOCKS_PER_SEC) << '\n';

    
    //Create index for entries
    thrust::device_vector<int> index(linecnt);
    index_filler fill_index((int *)thrust::raw_pointer_cast(index.data()));
    thrust::for_each(begin, begin+linecnt, fill_index);
    
    //Sort zoneid and index
    thrust::stable_sort_by_key(zoneid.begin(), zoneid.end(), index.begin());
    
    //Clean by checking standard deviation
    thrust::device_vector<int> device_linecnt(1);
    thrust::device_vector<int> new_flow(linecnt);
    device_linecnt[0] = linecnt;
    thrust::device_vector<float> global_std(1);
    global_std[0] = sqrt(result.variance_n());
    
    std_clean cleaner((int *)thrust::raw_pointer_cast(zoneid.data()), (int *)thrust::raw_pointer_cast(index.data()), (int *)thrust::raw_pointer_cast(flow.data()), (int *)thrust::raw_pointer_cast(new_flow.data()), (char *)thrust::raw_pointer_cast(flow_valid.data()), (int *)thrust::raw_pointer_cast(device_linecnt.data()), (float *)thrust::raw_pointer_cast(global_std.data()));
    thrust::for_each(begin, begin+linecnt, cleaner);
    cout << "Data cleaned: " << ((clock() - start)/(double)CLOCKS_PER_SEC) << '\n';

    
    //Convert values back to char
    //Laneid
    thrust::fill(laneid_text.begin(), laneid_text.end(), 0);
    gpu_itoa laneid_tochar((int *)thrust::raw_pointer_cast(laneid.data()), (char *)thrust::raw_pointer_cast(laneid_text.data()), (int *)thrust::raw_pointer_cast(column_width.data()));
    thrust::for_each(begin, begin + linecnt, laneid_tochar);
    
    //Flow
    thrust::fill(flow_text.begin(), flow_text.end(), 0);
    gpu_itoa flow_tochar((int *)thrust::raw_pointer_cast(flow.data()), (char *)thrust::raw_pointer_cast(flow_text.data()), (int *)thrust::raw_pointer_cast(&column_width[3]));
    thrust::for_each(begin, begin + linecnt, flow_tochar);
    
    //New Flow
    thrust::device_vector<char> new_flow_text(linecnt*column_width[3]);
    thrust::fill(new_flow_text.begin(), new_flow_text.end(), 0);
    gpu_itoa new_flow_tochar((int *)thrust::raw_pointer_cast(new_flow.data()), (char *)thrust::raw_pointer_cast(new_flow_text.data()), (int *)thrust::raw_pointer_cast(&column_width[3]));
    thrust::for_each(begin, begin + linecnt, new_flow_tochar);
    //thrust::host_vector<char> flow_host(linecnt*column_width[3]);
    //thrust::copy(flow_text.begin(), flow_text.end(), flow_host.begin());
    //cout << "Laneid and flow moved to host: " << ((clock() - start)/(double)CLOCKS_PER_SEC) << '\n';
    
    //Format output on GPU
    //Output includes laneid, previous flow, new flow, and validity, separated by commas
    thrust::device_vector<int> output_num_columns(1);
    output_num_columns[0] = 4;
    
    thrust::device_vector<char *> output_columns(output_num_columns[0]);
    output_columns[0] = thrust::raw_pointer_cast(laneid_text.data());
    output_columns[1] = thrust::raw_pointer_cast(flow_text.data());
    output_columns[2] = thrust::raw_pointer_cast(new_flow_text.data());
    output_columns[3] = thrust::raw_pointer_cast(flow_valid.data());
    
    thrust::device_vector<int> output_column_width(output_num_columns[0]);
    output_column_width[0] = column_width[0];
    output_column_width[1] = column_width[3];
    output_column_width[2] = column_width[3];
    output_column_width[3] = column_width[5];
    
    thrust::device_vector<int> output_size(1);
    output_size[0] = output_column_width[0] + output_column_width[1] + output_column_width[2] + output_column_width[3] + output_column_width.size();
    thrust::device_vector<char> output(linecnt * output_size[0]);
    thrust::fill(output.begin(), output.end(), ' ');
    gpu_output format_output((char **)thrust::raw_pointer_cast(output_columns.data()), (char *)thrust::raw_pointer_cast(output.data()), (int *)thrust::raw_pointer_cast(output_column_width.data()), (int *)thrust::raw_pointer_cast(output_size.data()), (int *)thrust::raw_pointer_cast(output_num_columns.data()));
    thrust::for_each(begin, begin + linecnt, format_output);
    thrust::device_vector<char> final_output(output.size());
    thrust::copy_if(output.begin(), output.end(), final_output.begin(), white_space());
    thrust::device_vector<char>::iterator output_end = thrust::find(final_output.begin(), final_output.end(), NULL);

    
    //Output cleaned data
    assert(SetCurrentDirectory("Cleaned") != 0);
    HANDLE outputFile = CreateFileA(argv[2], GENERIC_WRITE | GENERIC_READ, FILE_SHARE_WRITE, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
    assert(outputFile != INVALID_HANDLE_VALUE);
    
    HANDLE outputMap = CreateFileMapping(outputFile, NULL, PAGE_READWRITE, 0, output_end - final_output.begin(), NULL);
    if(outputMap == NULL)
        cout << GetLastError();        
 
    LPVOID outputMapView = MapViewOfFile(outputMap, FILE_MAP_WRITE, 0, 0, 0);
    
    if (outputMapView == NULL)
        cout << GetLastError();
    
    char *outputMapViewChar = (char *)outputMapView;
    thrust::copy(final_output.begin(), output_end, outputMapViewChar);

    cout << "Output written: " << ((clock() - start)/(double)CLOCKS_PER_SEC) << '\n';

    return 0;
}