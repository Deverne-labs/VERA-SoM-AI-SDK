/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
//USER include
#include "csr_offsets.h"
#include <stdlib.h>
#include <stdint.h>
#include "riscv.h"
#include "soc.h"
#include "bsp.h"
#include "plic.h"
#include "uart.h"
#include <math.h>
#include "print.h"
#include "clint.h"
#include "common.h"
#include "PiCamDriver.h"
//#include "apb3_cam.h"
#include "i2c.h"
//#include "i2cDemo.h"
extern "C" {
#include "dmasg.h"
}
//#include "axi4_hw_accel.h"

//Tinyml Header File
#include "intc.h"
#include "tinyml.h"
#include "ops/ops_api.h"

//Import TensorFlow lite libraries
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
//#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"
//#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/micro_time.h"
#include "platform/tinyml/profiler.h"


//Model data
#include "model/yolo_person_detect_model_data.h"

//Yolo layer
#include "model/yolo.h"

//VDMA driver
//#include "platform/vdma/vdma_driver.h"


#define SCALE 1
#define CLASSES 1
#define TOTAL_ANCHORS 3
#define NET_HEIGHT 96
#define NET_WIDTH 96
#define OBJECTNESS_THRESHOLD 0.25
#define IOU_THRESHOLD 0.5


//#define GAMMA_1


#define YOLO_PICO_INPUT_BYTES 96*96*4

#define FRAME_WIDTH     1280 //640
#define FRAME_HEIGHT    720 //360

#define OUT_FRAME_WIDTH  1280 //640
#define OUT_FRAME_HEIGHT 720 //360
#define OUT_FRAME_SIZE   OUT_FRAME_WIDTH*OUT_FRAME_HEIGHT*4

#define IN_FRAME_WIDTH  1280 //640
#define IN_FRAME_HEIGHT 720 //640
#define IN_FRAME_SIZE   IN_FRAME_WIDTH*IN_FRAME_HEIGHT*4

#define BITMAP_FRAME_WIDTH  1280 //640
#define BITMAP_FRAME_HEIGHT 720 //360
#define BITMAP_FRAME_SIZE   BITMAP_FRAME_WIDTH*BITMAP_FRAME_HEIGHT/8

#define LUT_TON_SIZE	1024



#define FRAME_W 1280 //640
#define FRAME_H 720  //640

#define ROI_W FRAME_W/2
#define ROI_H FRAME_H/2

#define FRAME_SIZE (FRAME_W*FRAME_H/4)
#define FRAME1_ADDR mem
#define FRAME2_ADDR mem+FRAME_SIZE



//Set to 4 for multi-buffering; Set to 1 for single buffering (shared for camera frame capture, display, and tinyML pre-processing input).
#define NUM_BUFFER   4
//Start address to be divided evenly by 8. Otherwise DMA tkeep might be shifted, not handled in display and hw_accel blocks.
//BUFFER_START_ADDR should not overlap with memory space allocated for RISC-V program (default.ld)
#define BUFFER0_START_ADDR        0x01800000
#define BUFFER1_START_ADDR        0x01600000
#define BUFFER2_START_ADDR        0x01400000
//Memory gap between BUFFER_START_ADDR and TINYML_INPUT_START_ADDR must sufficient to accommate NUM_BUFFER*FRAME_WIDTH*FRAME_HEIGHT*4 bytes data
#define TINYML_INPUT_START_ADDR   0x01300000 //0x01600000

#define BITMAP_START_ADDR         0x01200000
#define LUT_TON_START_ADDR        0x01100000

#define BBOX_MAX 16
//#define BBOX_CMD_ADDRESS   0x02000000
//The box buffer consist of command , data , and dummy data as padding to ensure that the total data transfer is always even
//DMA controller perform 128-bit word transfer, and our DMA channel will transfer 64-bit word to annotator
//Each transfer initiated from DMA controller will be 128-bit word, thus we need to ensure an even number of transfer
#define TOTAL_BOX_BUFFER   8 + (BBOX_MAX*8) + 8
//#define bbox_array ((volatile uint64_t*)BBOX_CMD_ADDRESS)

#define IMAGE_SIZE (IN_FRAME_WIDTH*IN_FRAME_HEIGHT)*4
#define IMAGE_CMD_OFFSET (0)
#define IMAGE_START_OFFSET (IMAGE_CMD_OFFSET + 8)
#define IMAGE_END_OFFSET (IMAGE_START_OFFSET + IMAGE_SIZE)
// The image consists of command, data and dummy data to ensure total data transfer is even , following DMA controller spec.
#define IMAGE_DUMMY_OFFSET (IMAGE_END_OFFSET + 8)
#define TOTAL_BUFFER_SIZE (IMAGE_DUMMY_OFFSET - IMAGE_CMD_OFFSET)


#define buffer0_array       ((uint32_t*)BUFFER0_START_ADDR)
#define buffer1_array       ((uint32_t*)BUFFER1_START_ADDR)
#define buffer2_array       ((uint32_t*)BUFFER2_START_ADDR)
#define tinyml_input_array ((volatile uint8_t*)TINYML_INPUT_START_ADDR)

//#define bitmap_array    ((volatile uint8_t*)BITMAP_START_ADDR)
//#define lut_ton_array   ((volatile uint16_t*)LUT_TON_START_ADDR)



uint8_t camera_buffer = 0;
uint8_t display_buffer = 0;
uint8_t next_display_buffer = 0;
uint8_t draw_buffer = 0;
uint8_t bbox_overlay_busy = 0;
uint8_t bbox_overlay_updated = 0;

uint8_t bitmap_buffer = 0;


namespace {
   tflite::ErrorReporter* error_reporter = nullptr;
   const tflite::Model* model = nullptr;
   tflite::MicroInterpreter* interpreter = nullptr;
   TfLiteTensor* model_input = nullptr;

   //Create an area of memory to use for input, output, and other TensorFlow
   //arrays. You'll need to adjust this by combiling, running, and looking
   //for errors.
   constexpr int kTensorArenaSize = 10000 * 1024;
   uint8_t tensor_arena[kTensorArenaSize];
   int total_output_layers = 0;
   float anchors[2][TOTAL_ANCHORS * 2] = {{115, 73, 119, 199, 242, 238}, {12, 18, 37, 49, 52, 132}};
}




u32 vdma_read_resetn(){
	return read_u32(IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_VDMA_RSTN_ADDR);
}
void vdma_write_resetn(int data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_VDMA_RSTN_ADDR);
}


u32 vdma_read_start_mm2s(){
	return read_u32(IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_START_VDMA_MM2S_ADDR);
}
void vdma_write_start_mm2s(int data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_START_VDMA_MM2S_ADDR);
}


u32 vdma_read_start_s2mm(){
	return read_u32(IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_START_VDMA_S2MM_ADDR);
}
void vdma_write_start_s2mm(int data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_START_VDMA_S2MM_ADDR);
}


u32 vdma_read_image_w(){
	return read_u32(IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_VDMA_IMAGE_W_ADDR);
}
void vdma_write_image_w(int data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_VDMA_IMAGE_W_ADDR);
}


u32 vdma_read_image_h(){
	return read_u32(IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_VDMA_IMAGE_H_ADDR);
}
void vdma_write_image_h(int data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_VDMA_IMAGE_H_ADDR);
}


u32 vdma_read_buffer0(){
	return read_u32(IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_ADDR_BUFFER0_ADDR);
}
void vdma_write_buffer0(uint32_t *framebuffer){
	write_u32((u32)framebuffer,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_ADDR_BUFFER0_ADDR);
}


u32 vdma_read_buffer1(){
	return read_u32(IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_ADDR_BUFFER1_ADDR);
}
void vdma_write_buffer1(uint32_t *framebuffer){
	write_u32((u32)framebuffer,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_ADDR_BUFFER1_ADDR);
}


u32 vdma_read_buffer2(){
	return read_u32(IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_ADDR_BUFFER2_ADDR);
}
void vdma_write_buffer2(uint32_t *framebuffer){
	write_u32((u32)framebuffer,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_ADDR_BUFFER2_ADDR);
}





#ifdef GAMMA_05
int gamma_tab[] ={   0,  16,  23,  28,  32,  36,  39,  42,  45,  48,  50,  53,  55,  58,  60,  62,
				64,  66,  68,  70,  71,  73,  75,  77,  78,  80,  81,  83,  84,  86,  87,  89,
				90,  92,  93,  94,  96,  97,  98, 100, 101, 102, 103, 105, 106, 107, 108, 109,
			   111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 125, 126, 127,
			   128, 129, 130, 131, 132, 133, 134, 135, 135, 136, 137, 138, 139, 140, 141, 142,
			   143, 144, 145, 145, 146, 147, 148, 149, 150, 151, 151, 152, 153, 154, 155, 156,
			   156, 157, 158, 159, 160, 160, 161, 162, 163, 164, 164, 165, 166, 167, 167, 168,
			   169, 170, 170, 171, 172, 173, 173, 174, 175, 176, 176, 177, 178, 179, 179, 180,
			   181, 181, 182, 183, 183, 184, 185, 186, 186, 187, 188, 188, 189, 190, 190, 191,
			   192, 192, 193, 194, 194, 195, 196, 196, 197, 198, 198, 199, 199, 200, 201, 201,
			   202, 203, 203, 204, 204, 205, 206, 206, 207, 208, 208, 209, 209, 210, 211, 211,
			   212, 212, 213, 214, 214, 215, 215, 216, 217, 217, 218, 218, 219, 220, 220, 221,
			   221, 222, 222, 223, 224, 224, 225, 225, 226, 226, 227, 228, 228, 229, 229, 230,
			   230, 231, 231, 232, 233, 233, 234, 234, 235, 235, 236, 236, 237, 237, 238, 238,
			   239, 240, 240, 241, 241, 242, 242, 243, 243, 244, 244, 245, 245, 246, 246, 247,
			   247, 248, 248, 249, 249, 250, 250, 251, 251, 252, 252, 253, 253, 254, 254, 255
};
#endif
#ifdef GAMMA_1
int gamma_tab[] ={
     0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
    16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
    32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
    48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
    64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
    80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
    96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
   112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
   128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
   144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
   160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
   176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
   192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
   208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
   224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
   240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255
};
#endif

#ifdef GAMMA_15
int gamma_tab[] ={
     0,   0,   0,   0,   1,   1,   1,   1,   1,   2,   2,   2,   3,   3,   3,   4,
     4,   4,   5,   5,   6,   6,   6,   7,   7,   8,   8,   9,   9,  10,  10,  11,
    11,  12,  12,  13,  14,  14,  15,  15,  16,  16,  17,  18,  18,  19,  20,  20,
    21,  21,  22,  23,  23,  24,  25,  26,  26,  27,  28,  28,  29,  30,  31,  31,
    32,  33,  34,  34,  35,  36,  37,  37,  38,  39,  40,  41,  41,  42,  43,  44,
    45,  46,  46,  47,  48,  49,  50,  51,  52,  53,  53,  54,  55,  56,  57,  58,
    59,  60,  61,  62,  63,  64,  65,  65,  66,  67,  68,  69,  70,  71,  72,  73,
    74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  88,  89,  90,
    91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 102, 103, 104, 105, 106, 107,
   108, 109, 110, 112, 113, 114, 115, 116, 117, 119, 120, 121, 122, 123, 124, 126,
   127, 128, 129, 130, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 144, 145,
   146, 147, 149, 150, 151, 152, 154, 155, 156, 158, 159, 160, 161, 163, 164, 165,
   167, 168, 169, 171, 172, 173, 174, 176, 177, 178, 180, 181, 182, 184, 185, 187,
   188, 189, 191, 192, 193, 195, 196, 197, 199, 200, 202, 203, 204, 206, 207, 209,
   210, 211, 213, 214, 216, 217, 218, 220, 221, 223, 224, 226, 227, 228, 230, 231,
   233, 234, 236, 237, 239, 240, 242, 243, 245, 246, 248, 249, 251, 252, 254, 255
};
#endif

#ifdef GAMMA_02
int gamma_tab[] ={
     0,  84,  97, 105, 111, 116, 120, 124, 128, 131, 133, 136, 138, 141, 143, 145,
   147, 148, 150, 152, 153, 155, 156, 158, 159, 160, 162, 163, 164, 165, 166, 167,
   168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 179, 180, 181, 182,
   183, 183, 184, 185, 186, 186, 187, 188, 188, 189, 190, 190, 191, 192, 192, 193,
   193, 194, 195, 195, 196, 196, 197, 197, 198, 199, 199, 200, 200, 201, 201, 202,
   202, 203, 203, 204, 204, 205, 205, 206, 206, 207, 207, 208, 208, 208, 209, 209,
   210, 210, 211, 211, 211, 212, 212, 213, 213, 214, 214, 214, 215, 215, 216, 216,
   216, 217, 217, 217, 218, 218, 219, 219, 219, 220, 220, 220, 221, 221, 221, 222,
   222, 223, 223, 223, 224, 224, 224, 225, 225, 225, 226, 226, 226, 227, 227, 227,
   227, 228, 228, 228, 229, 229, 229, 230, 230, 230, 231, 231, 231, 231, 232, 232,
   232, 233, 233, 233, 233, 234, 234, 234, 235, 235, 235, 235, 236, 236, 236, 237,
   237, 237, 237, 238, 238, 238, 238, 239, 239, 239, 239, 240, 240, 240, 240, 241,
   241, 241, 241, 242, 242, 242, 242, 243, 243, 243, 243, 244, 244, 244, 244, 245,
   245, 245, 245, 246, 246, 246, 246, 246, 247, 247, 247, 247, 248, 248, 248, 248,
   248, 249, 249, 249, 249, 250, 250, 250, 250, 250, 251, 251, 251, 251, 252, 252,
   252, 252, 252, 253, 253, 253, 253, 253, 254, 254, 254, 254, 254, 255, 255, 255
};
#endif







void tinyml_init() {
   //Set up logging
   static tflite::MicroErrorReporter micro_error_reporter;
   error_reporter = &micro_error_reporter;

   //Map the model into a usable data structure
   model = tflite::GetModel(yolo_person_detect_model_data);




   //AllOpsResolver may be used for generalization
   static tflite::AllOpsResolver resolver;
   tflite::MicroOpResolver* op_resolver = nullptr;
   op_resolver = &resolver;

   //Build an interpreter to run the model

   FullProfiler prof;
   static tflite::MicroInterpreter static_interpreter(
      model, *op_resolver, tensor_arena, kTensorArenaSize,
      error_reporter, nullptr); //Without profiler
      //error_reporter, &prof); //With profiler
   interpreter = &static_interpreter;
   prof.setInterpreter(interpreter);
   prof.setDump(false);
   
   //Allocate memory from the tensor_arena for the model's tensors
   TfLiteStatus allocate_status = interpreter->AllocateTensors();


   //Assign model input buffer (tensor) to pointer
   model_input = interpreter->input(0);

   //Print loaded model input and output shape
   total_output_layers = interpreter->outputs_size();

}


static void flush_data_cache(){
   asm(".word(0x500F)");
}


void send_dma(u32 channel, u32 port, u32 addr, u32 size, int interrupt, int wait, int self_restart) {
   dmasg_input_memory(DMASG_BASE, channel, addr, 16);
   dmasg_output_stream(DMASG_BASE, channel, port, 0, 0, 1);
   
   if(interrupt) {
      dmasg_interrupt_config(DMASG_BASE, channel, DMASG_CHANNEL_INTERRUPT_CHANNEL_COMPLETION_MASK);
   }
   
   /*
   if(self_restart) {
      dmasg_direct_start(DMASG_BASE, channel, size, 1);
   } else {
      dmasg_direct_start(DMASG_BASE, channel, size, 0);
   }*/
   dmasg_direct_start(DMASG_BASE, channel, size, self_restart);
   
   if(wait) {
      while(dmasg_busy(DMASG_BASE, channel));
      flush_data_cache();
   }
}

void recv_dma(u32 channel, u32 port, u32 addr, u32 size, int interrupt, int wait, int self_restart) {
   dmasg_input_stream(DMASG_BASE, channel, port, 1, 0);
   dmasg_output_memory(DMASG_BASE, channel, addr, 16);
   
   if(interrupt){
      dmasg_interrupt_config(DMASG_BASE, channel, DMASG_CHANNEL_INTERRUPT_CHANNEL_COMPLETION_MASK);
   }

   /*if(self_restart) {
      dmasg_direct_start(DMASG_BASE, channel, size, 1);
   } else {
      dmasg_direct_start(DMASG_BASE, channel, size, 0);
   }*/
   dmasg_direct_start(DMASG_BASE, channel, size, self_restart);

   if(wait){
      while(dmasg_busy(DMASG_BASE, channel));
      flush_data_cache();
   }
}



void reset_bitmap(){
   uint32_t *buffer = (uint32_t *)BITMAP_START_ADDR;
   for(int i = 0 ; i < BITMAP_FRAME_HEIGHT; i++){
      for(int j = 0 ; j < BITMAP_FRAME_WIDTH/32; j++){
         buffer[i*(BITMAP_FRAME_WIDTH/32) + j] = 0x00;
      }
   }
}

/*void write_gamma_lut(){
   uint8_t *buffer = (uint8_t *)LUT_TON_START_ADDR;
   int ctr = 0;
   for(int i = 0 ; i < LUT_TON_SIZE; i++){
		buffer[i] = (uint8_t)ctr;//gamma_tab[i];
		if(i%4 == 0 & i != 0)
			ctr++;
   }
}*/

/*
void print_img_downscaled(volatile u32* buf){
   int px;
   int ds_width = 96/4;
   for (int y=0; y<96; y++) {
	 for (int x=0; x<96/4; x++) {
	   px =  (buf[y*ds_width + x] & 0x000000FF)>>0;
	   MicroPrintf("%d ",px);
	   px =  (buf[y*ds_width + x] & 0x0000FF00)>>8;
	   MicroPrintf("%d ",px);
	   px =  (buf[y*ds_width + x] & 0x00FF0000)>>16;
	   MicroPrintf("%d ",px);
	   px =  (buf[y*ds_width + x] & 0xFF000000)>>24;
	   MicroPrintf("%d ",px);
	 }
	 MicroPrintf("\n\r ");
   }
}*/


void init() {
   /************************************************************SETUP PICAM************************************************************/

  // MicroPrintf("Camera Setting...");
   //Camera I2C configuration
   mipi_i2c_init();
   //PiCam_init();


  // MicroPrintf("Done\n\r");

   //Indicate camera configuration done
  // EXAMPLE_APB3_REGW(EXAMPLE_APB3_SLV, EXAMPLE_APB3_SLV_REG1_OFFSET, 0x00000001);
  // MicroPrintf("Done\n\r");

   //SET camera pre-processing RGB gain value
  // Set_RGBGain(1,5,3,4);

  // MicroPrintf("Done\n\r");



   //vdma_write_resetn(0);
   //vdma_write_resetn(1);
   write_u32(0,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_VDMA_RSTN_ADDR);
   write_u32(1,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_REG_VDMA_RSTN_ADDR);

   //reset rx pipeline
   write_u32(0,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_RSTN_ADDR);
   //msDelay(200);
   write_u32(1,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_RSTN_ADDR);


   //Config RX pipeline
   write_u32(1280 ,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_WIDTH_ADDR          );
   write_u32(720  ,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_HEIGHT_ADDR         );
   write_u32(0    ,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_GAMMA_EN_UPDATE_ADDR);
   write_u32(1    ,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_GAMMA_EN_BYPASS_ADDR);
   write_u32(0x200,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_UNBAYER_R_GAIN_ADDR );
   write_u32(0x180,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_UNBAYER_G_GAIN_ADDR );
   write_u32(0x208,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_UNBAYER_B_GAIN_ADDR );

   write_u32(280,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_DOWNSCALE_OFFSET_X_ADDR );
   write_u32(7,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_DOWNSCALE_RATIO_ADDR );
   write_u32(945,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_DOWNSCALE_IMAGE_IN_X_MAX_ADDR );
   write_u32(665,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_DOWNSCALE_IMAGE_IN_Y_MAX_ADDR );

   //write_u32(1    ,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_START_ADDR );

   /*************************************************************SETUP DMA*************************************************************/

  // MicroPrintf("DMA Setting...");
   dma_init();

  // dmasg_priority(DMASG_BASE, DMASG_HW_ACCEL_MM2S_CHANNEL, 0, 0);
  // dmasg_priority(DMASG_BASE, DMASG_HW_ACCEL_S2MM_CHANNEL, 0, 0);
  // dmasg_priority(DMASG_BASE, DMASG_DISPLAY_MM2S_CHANNEL,  3, 0);
   dmasg_priority(DMASG_BASE, DMASG_BITMAP_MM2S_CHANNEL,    1, 0);
   dmasg_priority(DMASG_BASE, DMASG_LUT_TON_MM2S_CHANNEL,   0, 0);
   dmasg_priority(DMASG_BASE, DMASG_IMG_96x96_MM2S_CHANNEL, 2, 0);
  // dmasg_priority(DMASG_BASE, DMASG_CAM_S2MM_CHANNEL,      0, 0);

  // MicroPrintf("Done\n\r");

   /*************************************************************SETUP VDMA*************************************************************/

   vdma_write_image_w(FRAME_WIDTH);
   vdma_write_image_h(FRAME_HEIGHT);

   vdma_write_buffer0(buffer0_array);
   vdma_write_buffer1(buffer1_array);
   vdma_write_buffer2(buffer2_array);


   recv_dma(DMASG_IMG_96x96_MM2S_CHANNEL, DMASG_IMG_96x96_MM2S_PORT, TINYML_INPUT_START_ADDR, YOLO_PICO_INPUT_BYTES, 0, 0, 1);

   vdma_write_start_s2mm(1);

   //MicroPrintf("Readback:\n\r");
   //MicroPrintf("Start s2mm : %d\n\r",int(vdma_read_start_s2mm()));
   //MicroPrintf("Resetn s2mm : %d\n\r",int(vdma_read_resetn()));

   //msDelay(200);

	PCam5C_init();

	PCam5C_config();
	PCam5C_set_awb();


   /***********************************************************TRIGGER DISPLAY*******************************************************/

   //MicroPrintf("Initialize display memory content...");

   //Initialize test image in buffer_array (default buffer 0) 
   //init_image();
   //patern_bitmap();
   //patern_lut_ton();
   //write_gamma_lut();

//   send_dma(DMASG_LUT_TON_MM2S_CHANNEL, DMASG_LUT_TON_MM2S_PORT,LUT_TON_START_ADDR, LUT_TON_SIZE, 0, 0, 1);


   //MicroPrintf("Done\n\r");
   //Initialize bbox_overlay_buffer - Trigger DMA for initialized bbox_overlay_buffer content to display annotator module!!!
   //MicroPrintf("Initialize Bbox to invalid ...");
   //init_bbox();
   //MicroPrintf("Done\n\r");
   //Trigger display DMA once then the rest handled by interrupt sub-rountine

   
   /*********************************************************TRIGGER CAMERA CAPTURE*****************************************************/
   
   //SELECT RGB or grayscale output from camera pre-processing block.
  // EXAMPLE_APB3_REGW(EXAMPLE_APB3_SLV, EXAMPLE_APB3_SLV_REG3_OFFSET, 0x00000000);   //RGB

   //Trigger camera DMA once then the rest handled by interrupt sub-rountine
   //MicroPrintf("Trigger camera DMA...");
   //msDelay(5000);


//recv_dma(DMASG_CAM_S2MM_CHANNEL, DMASG_CAM_S2MM_PORT, buf(display_buffer), 640*640*4, 1, 0, 1); //buf_offset(camera_buffer, IMAGE_START_OFFSET)
   //cam_s2mm_active = 1;



   //Indicate start of S2MM DMA to camera building block via APB3 slave
   //EXAMPLE_APB3_REGW(EXAMPLE_APB3_SLV, EXAMPLE_APB3_SLV_REG4_OFFSET, 0x00000001);
   //EXAMPLE_APB3_REGW(EXAMPLE_APB3_SLV, EXAMPLE_APB3_SLV_REG4_OFFSET, 0x00000000);

   //Trigger storage of one captured frame via APB3 slave
   //EXAMPLE_APB3_REGW(EXAMPLE_APB3_SLV, EXAMPLE_APB3_SLV_REG2_OFFSET, 0x00000003);
   //EXAMPLE_APB3_REGW(EXAMPLE_APB3_SLV, EXAMPLE_APB3_SLV_REG2_OFFSET, 0x00000000);

   //MicroPrintf("Done\n\r");

   /***********************************************************TFLITE-MICRO TINYML*******************************************************/

  // MicroPrintf("TinyML Setup...");
   tinyml_init();
  // MicroPrintf("Done\n\r");
}

void draw_boxes(box* boxes,int total_boxes){
   //To store coordinates information
   float min_val = 0.00;
   float max_val = 1.00;
   uint16_t x_min[BBOX_MAX];
   uint16_t x_max[BBOX_MAX];
   uint16_t y_min[BBOX_MAX];
   uint16_t y_max[BBOX_MAX];
   uint64_t box_coordinates;
   float objectness_tresh=0.39;
   int count_boxes=0;

   uint8_t *buffer_bm = (uint8_t *)BITMAP_START_ADDR;
   uint8_t bm_elt = 0;

   reset_bitmap();
   //patern_bitmap();

   for (int i=0; i<total_boxes && i<BBOX_MAX; i++){
      if(boxes[i].x_min < min_val || boxes[i].y_min < min_val || boxes[i].x_max < min_val|| boxes[i].y_max <min_val || boxes[i].x_min > max_val || boxes[i].y_min > max_val ||  i>total_boxes || boxes[i].objectness < objectness_tresh ){
         x_min[i] = 0;
         y_min[i] = 0;
         x_max[i] = 0;
         y_max[i] = 0;
      }
      else {
         x_min[i] = (boxes[i].x_min)*IN_FRAME_WIDTH;
         y_min[i] = (boxes[i].y_min)*IN_FRAME_HEIGHT;
         x_max[i] = (boxes[i].x_max)*IN_FRAME_WIDTH;
         y_max[i] = (boxes[i].y_max)*IN_FRAME_HEIGHT;

         if(x_max[i] > OUT_FRAME_WIDTH){
            x_max[i] = (OUT_FRAME_WIDTH-1);
         }
         if(y_max[i] > OUT_FRAME_HEIGHT){
            y_max[i] = (OUT_FRAME_HEIGHT-1);
         }
         MicroPrintf("Box %d : %d %d %d %d \n\r",count_boxes,x_min[i],x_max[i],y_min[i],y_max[i]);


         for(uint16_t coord_y=y_min[i]; coord_y<=y_max[i]; coord_y++){
             for(uint16_t coord_x=x_min[i]; coord_x<=x_max[i]; coord_x++){
            	if((coord_y==y_min[i]) || (coord_y==y_max[i])) {
				   bm_elt = buffer_bm[coord_x/8 + coord_y*(BITMAP_FRAME_WIDTH/8)];
				   bm_elt = bm_elt | (0x1 << (coord_x%8));
				   buffer_bm[coord_x/8 + coord_y*(BITMAP_FRAME_WIDTH/8)] = bm_elt;
            	}
            	else if((coord_x==x_min[i]) || (coord_x==x_max[i])){
 				   bm_elt = buffer_bm[coord_x/8 + coord_y*(BITMAP_FRAME_WIDTH/8)];
 				   bm_elt = bm_elt | (0x1 << (coord_x%8));
 				   buffer_bm[coord_x/8 + coord_y*(BITMAP_FRAME_WIDTH/8)] = bm_elt;
            	}
            }
         }

         count_boxes++;
      }
   }


   MicroPrintf("Total Boxes : %d\n\r",count_boxes);
   //bbox_overlay_updated=1;
}



#include "gpio.h"
#include "soc.h"
#include "compatibility.h"
#include "i2c.h"
#include "common.h"
#include "fb.h"

//uint32_t framebuffer[1*1*3];


/*#ifdef SYSTEM_GPIO_0_IO_CTRL
    #define GPIO0       SYSTEM_GPIO_0_IO_CTRL
#endif*/


//#define CLOK_15x1503_MMAP_LEN 64
//External Clock generator utils
//#define CLOCK_GEN_BAR 0x68

#define APB3_REGW(addr, offset, data) \
   write_u32(data, addr+offset)

#define APB3_REGR(addr, offset) \
   read_u32(addr+offset)

//u16* raw_video_frmb    = (u16*)0x41000000; //video frame buffer input


//#define I2C_CTRL_HZ SYSTEM_CLINT_HZ


/*void busy_wait(int delay){
   msDelay(delay);
}*/



//"h_active", "h_blanking", "h_sync_offset", "h_sync_width", "v_active", "v_blanking", "v_sync_offset", "v_sync_width"

#define PRECALCUL_VTG


void config_pipe(){

  // uint32_t fifo_level = 1024;
   //uint32_t fifo_level_i = 1024;
   //uint32_t fifo_level_out = 1024;

#ifndef PRECALCUL_VTG
   int hres = 1280;
   int hsync_start = 1280 + 110;
   int hsync_end = 1280+ 110 + 40;
   int hscan = 1280 + 370 - 1;

   int vres = 720;
   int vsync_start = 720 + 5;
   int vsync_end = 720 + 5 + 5;
   int vscan = 720 + 30 - 1;
#endif

     // write_u32(1280,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_WIDTH_ADDR);
     // write_u32(720,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_HEIGHT_ADDR);


   //VTG
#ifdef PRECALCUL_VTG
  /*write_u32(1280,IO_APB_SLAVE_3_INPUT+CSR_VTG_HRES_ADDR);
  write_u32(1390,IO_APB_SLAVE_3_INPUT+CSR_VTG_HSYNC_START_ADDR);
  write_u32(1430,IO_APB_SLAVE_3_INPUT+CSR_VTG_HSYNC_END_ADDR);
  write_u32(1649,IO_APB_SLAVE_3_INPUT+CSR_VTG_HSCAN_ADDR);

  write_u32(720,IO_APB_SLAVE_3_INPUT+CSR_VTG_VRES_ADDR);
  write_u32(725,IO_APB_SLAVE_3_INPUT+CSR_VTG_VSYNC_START_ADDR);
  write_u32(730,IO_APB_SLAVE_3_INPUT+CSR_VTG_VSYNC_END_ADDR);
  write_u32(749,IO_APB_SLAVE_3_INPUT+CSR_VTG_VSCAN_ADDR);*/
#else
   write_u32(hres,IO_APB_SLAVE_3_INPUT+CSR_VTG_HRES_ADDR);
   write_u32(hsync_start,IO_APB_SLAVE_3_INPUT+CSR_VTG_HSYNC_START_ADDR);
   write_u32(hsync_end,IO_APB_SLAVE_3_INPUT+CSR_VTG_HSYNC_END_ADDR);
   write_u32(hscan,IO_APB_SLAVE_3_INPUT+CSR_VTG_HSCAN_ADDR);

   write_u32(vres,IO_APB_SLAVE_3_INPUT+CSR_VTG_VRES_ADDR);
   write_u32(vsync_start,IO_APB_SLAVE_3_INPUT+CSR_VTG_VSYNC_START_ADDR);
   write_u32(vsync_end,IO_APB_SLAVE_3_INPUT+CSR_VTG_VSYNC_END_ADDR);
   write_u32(vscan,IO_APB_SLAVE_3_INPUT+CSR_VTG_VSCAN_ADDR);
#endif

   //send_dma(DMASG_LUT_TON_MM2S_CHANNEL, DMASG_LUT_TON_MM2S_PORT,LUT_TON_START_ADDR, 1024, 0, 0, 1);
   //write_u32(0    ,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_GAMMA_EN_UPDATE_ADDR);
   //write_u32(1    ,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_GAMMA_EN_BYPASS_ADDR);

   //start tx pipeline
   write_u32(1,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_START_ADDR);

   vdma_write_start_mm2s(1); //dma_video_out_execution(framebuffer);

   send_dma(DMASG_BITMAP_MM2S_CHANNEL, DMASG_BITMAP_MM2S_PORT,BITMAP_START_ADDR, BITMAP_FRAME_SIZE, 0, 0, 1);

   msDelay(500);
   /*fifo_level = 0;fifo_level_i =0;fifo_level_out = 0;

   while(fifo_level_out < 500 ){
      fifo_level_out = fb_fifo_out_level_read();
      MicroPrintf("fifo_level : %d \n\r",fifo_level_out);
      msDelay(100);
   }*/

   //enable vtg
   write_u32(1,IO_APB_SLAVE_3_INPUT+CSR_VTG_ENABLE_ADDR);

}

//END USER include

void main() {
   //u32 rdata;
   init();
   config_pipe();
   TfLiteStatus invoke_status;
#ifndef ENABLE_IA
   while(1) {

      /***********************************************HW ACCELERATOR - TINYML PRE-PROCESSING********************************************/
      //flush_data_cache();

      /*******************************************************TINYML INFERENCE**********************************************************/

      //MicroPrintf("TinyML Inference...");

      //Copy test image to tflite model input.
      for (unsigned int i = 0; i < YOLO_PICO_INPUT_BYTES; ++i)
         model_input->data.int8[i] = tinyml_input_array[i] - 128; //Input normalization: From range [0,255] to [-128,127]

      //Perform inference
      //timerCmp0 = clint_getTime(BSP_CLINT);
      invoke_status = interpreter->Invoke();
      //timerCmp1 = clint_getTime(BSP_CLINT);

      /*if (invoke_status != kTfLiteOk) {
        MicroPrintf("Invoke failed on data\n\r");
      }*/
      //MicroPrintf("Done\n\n\r");

      //Yolo layer
      //MicroPrintf("Pass data to Yolo layer...");

      layer* yolo_layers = (layer*)calloc(total_output_layers, sizeof(layer));
      for (int i = 0; i < total_output_layers; ++i) {
         yolo_layers[i].channels = interpreter->output(i)->dims->data[0];
         yolo_layers[i].height = interpreter->output(i)->dims->data[1];
         yolo_layers[i].width = interpreter->output(i)->dims->data[2];
         yolo_layers[i].classes = CLASSES;
         yolo_layers[i].boxes_per_scale = interpreter->output(i)->dims->data[3] / (5 + yolo_layers[i].classes);
         yolo_layers[i].total_anchors = TOTAL_ANCHORS;
         yolo_layers[i].scale = SCALE;
         yolo_layers[i].anchors = anchors[i];

         int total = (
            interpreter->output(i)->dims->data[0] * interpreter->output(i)->dims->data[1] * interpreter->output(i)->dims->data[2] * interpreter->output(i)->dims->data[3]
         );

         yolo_layers[i].outputs = (float*)calloc(total, sizeof(float));
         TfLiteAffineQuantization params = *(static_cast<TfLiteAffineQuantization *>(interpreter->output(i)->quantization.params));

         for (int j = 0; j < total; ++j)
            yolo_layers[i].outputs[j] = ((float)interpreter->output(i)->data.int8[j] - params.zero_point->data[0]) * params.scale->data[0];
      }

      //MicroPrintf("Done\n\r");

      int total_boxes = 0;

      //MicroPrintf("Yolo layer inference...");
      box* boxes = perform_inference(yolo_layers, total_output_layers, &total_boxes, model_input->dims->data[1], model_input->dims->data[2], OBJECTNESS_THRESHOLD, IOU_THRESHOLD);

      //Pass bounding boxes info to annotator
      draw_boxes(boxes,total_boxes);


      //Clear all the memory allocation content
      for (int i = 0; i < total_output_layers; ++i) {
        free(yolo_layers[i].outputs);
      }
      for (int i = 0; i < total_boxes; ++i) {
       free(boxes[i].class_probabilities);
      }
      free(yolo_layers);
      free(boxes);

      //Switch draw buffer to latest complete frame
      //draw_buffer = next_display_buffer;
   }
#endif
}
