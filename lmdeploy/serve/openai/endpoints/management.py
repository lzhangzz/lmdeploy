# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations

import os
from http import HTTPStatus

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response

from lmdeploy.serve.openai.errors import create_error_response
from lmdeploy.serve.openai.protocol import AbortRequest


def register(router: APIRouter, server_context) -> None:

    @router.get('/health')
    async def health() -> JSONResponse:
        """Health check."""
        monitor = server_context.health_monitor
        if monitor is None:
            data = dict(status='unhealthy',
                        message='Engine health monitor is not initialized.')
            return JSONResponse(jsonable_encoder(data),
                                status_code=HTTPStatus.SERVICE_UNAVAILABLE)
        data = monitor.snapshot()
        if data['status'] == 'unhealthy':
            data = await monitor.refresh_snapshot()
        status_code = HTTPStatus.OK if data['status'] == 'healthy' else HTTPStatus.SERVICE_UNAVAILABLE
        return JSONResponse(jsonable_encoder(data), status_code=status_code)

    @router.get('/terminate')
    async def terminate():
        """Terminate server."""
        import signal

        if not server_context.allow_terminate_by_client:
            return create_error_response(
                HTTPStatus.BAD_REQUEST,
                'The server can not be terminated. Please add --allow-terminate-by-client when start the server.'
            )
        os.kill(os.getpid(), signal.SIGTERM)
        return Response(status_code=200)

    @router.post('/abort_request')
    async def abort_request(request: AbortRequest,
                            raw_request: Request = None):
        """Abort an ongoing request."""
        if not server_context.enable_abort_handling:
            return Response(
                status_code=501,
                content=
                'This server does not support abort requests. Enable with --enable-abort-handling flag.'
            )

        if request.abort_all:
            await server_context.async_engine.stop_all_session()
        else:
            session = server_context.find_session(request.session_id)
            if session is None:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    f'Session {request.session_id} not found.')
            await session.async_abort()
            session_mgr = server_context.session_manager
            session_mgr.remove(session)
        return Response(status_code=200)
