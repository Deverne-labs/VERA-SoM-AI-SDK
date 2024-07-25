
#include "platform/csr_offsets.h"

/*
u32 fb_idle_upscale_read(){
	return read_u32(IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_IDLE_UPSCALE_ADDR);
}

u32 fb_ready_upscale_read(){
	return read_u32(IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_READY_UPSCALE_ADDR);
}

u32 fb_fifo_level_read(){
	return read_u32(IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_FIFO_LEVEL_ADDR);
}

u32 fb_fifo_i_level_read(){
	return read_u32(IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_FIFO_I_LEVEL_ADDR);
}

u32 fb_fifo_bitmap_level_read(){
	return read_u32(IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_FIFO_BITMAP_LEVEL_ADDR);
}*/

u32 fb_fifo_out_level_read(){
	return read_u32(IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_FIFO_OUT_LEVEL_ADDR);
}


/*
void fb_str_pipe_write(u32 data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_FB_STR_PIPE_ADDR);
}

void fb_enable_pipeline_write(u32 data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_FB_ENABLE_PIPELINE_ADDR);
} */

void fb_start_txpipeline_write(u32 data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_START_ADDR);
}

void fb_rst_pipeline_write(u32 data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_RSTN_ADDR);
}

void fb_pipeline_hres_write(u32 data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_WIDTH_ADDR);
}

void fb_pipeline_vres_write(u32 data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_HEIGHT_ADDR);
}

/*void fb_bypass_upscale_write(u32 data){
	write_u32(data,IO_APB_SLAVE_3_INPUT+CSR_TXPIPELINE_BYPASS_UPSCALE_ADDR);
}*/
